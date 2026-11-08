"""
2D top-down car racing environment (pygame) for programmatic control
(genetic algorithms, neuroevolution, RL, ...).

Usage from code:

    from car_env import CarEnv
    env = CarEnv(n_cars=50, difficulty=0.5, render_mode="human", seed=0)
    obs = env.reset()                       # obs["image"]: (n_cars, 128, 128, 3) uint8
    while not env.all_done:
        actions = np.zeros((env.n_cars, 2))  # [steer, throttle] in [-1, 1]
        obs, rewards, dones, info = env.step(actions)
    print(env.fitness)                      # progress per car (1.0 = one lap)
    env.close()

Run directly:
    python car_env.py                 # drive one car with the arrow keys
    python car_env.py --demo          # 30 cars driven by a simple ray-based controller
    python car_env.py --difficulty 0.9 --seed 3
"""

import argparse
import math
import os

import numpy as np
import pygame

OBS_SIZE = 128

GRASS = (46, 120, 50)
ROAD = (90, 90, 95)
CURB = (230, 230, 230)
LINE = (240, 210, 60)


class Track:
    """Closed road generated as a star-shaped curve r(theta) = R * (1 + sum of harmonics).

    difficulty in [0, 1]: 0 -> wide, gentle turns; 1 -> many tight turns.
    """

    def __init__(self, width=1200, height=800, road_width=70, difficulty=0.5, seed=None, n_points=1200):
        self.width, self.height = width, height
        self.road_width = road_width
        rng = np.random.default_rng(seed)

        n_harmonics = 2 + int(round(difficulty * 7))
        amp_total = 0.15 + 0.5 * difficulty
        ks = np.arange(2, 2 + n_harmonics)
        amps = rng.uniform(0.3, 1.0, n_harmonics) / ks ** 0.6
        amps *= amp_total / amps.sum()
        phases = rng.uniform(0, 2 * np.pi, n_harmonics)

        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        r = 1.0 + (amps[:, None] * np.sin(ks[:, None] * theta[None, :] + phases[:, None])).sum(0)
        margin = road_width
        rx = width / 2 - margin
        ry = height / 2 - margin
        r = r / r.max()
        r = np.maximum(r, 0.35)
        pts = np.stack([width / 2 + rx * r * np.cos(theta), height / 2 + ry * r * np.sin(theta)], 1)

        # Resample so points are evenly spaced along the arc length.
        seg = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        self.length = cum[-1]
        s = np.linspace(0, self.length, n_points, endpoint=False)
        closed = np.vstack([pts, pts[:1]])
        self.center = np.stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])], 1)
        self.n = n_points
        self.spacing = self.length / n_points

        self._build_surface()

    def _build_surface(self):
        surf = pygame.Surface((self.width, self.height))
        surf.fill(GRASS)
        hw = self.road_width / 2
        for p in self.center:
            pygame.draw.circle(surf, CURB, p, hw + 4)
        for p in self.center:
            pygame.draw.circle(surf, ROAD, p, hw)
        for i in range(0, self.n, 12):
            a, b = self.center[i], self.center[(i + 5) % self.n]
            pygame.draw.line(surf, LINE, a, b, 2)
        # start / finish line
        p0, p1 = self.center[0], self.center[1]
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        nrm = np.array([-d[1], d[0]])
        pygame.draw.line(surf, (255, 255, 255), p0 - nrm * hw, p0 + nrm * hw, 4)

        self.surface = surf
        rgb = pygame.surfarray.array3d(surf)  # (W, H, 3)
        self.rgb = np.ascontiguousarray(rgb.transpose(1, 0, 2))  # (H, W, 3)
        self.on_road = np.all(self.rgb != np.array(GRASS, dtype=np.uint8), axis=2)
        curb = np.all(self.rgb == np.array(CURB, dtype=np.uint8), axis=2)
        self.on_road &= ~curb  # touching the curb = crash

    def start_pose(self):
        p0, p1 = self.center[0], self.center[3]
        return p0.copy(), math.atan2(p1[1] - p0[1], p1[0] - p0[0])


class CarEnv:
    """Multiple independent cars (no car-car collisions) on the same track.

    Action per car: [steer, throttle], both in [-1, 1].
    Observation dict:
        image : (n_cars, 128, 128, 3) uint8, car-centric view, car always points up
        rays  : (n_cars, n_rays) float in [0, 1], distance to road edge / ray_length
        speed : (n_cars,) float in [0, 1]
    """

    def __init__(self, n_cars=1, difficulty=0.5, seed=None, render_mode=None, fps=60,
                 road_width=70, max_steps=3000, stall_steps=150, view_scale=2.0,
                 n_rays=7, ray_length=250.0, compute_images=True,
                 max_speed=7.0, accel=0.25, drag=0.02, min_turn_radius=35.0):
        self.n_cars = n_cars
        self.difficulty = difficulty
        self.seed = seed
        self.render_mode = render_mode
        self.fps = fps
        self.road_width = road_width
        self.max_steps = max_steps
        self.stall_steps = stall_steps
        self.view_scale = view_scale  # world pixels per observation pixel
        self.n_rays = n_rays
        self.ray_length = ray_length
        self.compute_images = compute_images
        self.max_speed, self.accel, self.drag = max_speed, accel, drag
        self.min_turn_radius = min_turn_radius
        self.ray_angles = np.linspace(-np.pi / 2, np.pi / 2, n_rays)

        if render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.screen = None
        self.clock = None
        self.font = None
        self.track = Track(road_width=road_width, difficulty=difficulty, seed=seed)
        self._car_sprites = None

        # observation sampling grid (car frame: x right, y forward)
        c = (np.arange(OBS_SIZE) - OBS_SIZE / 2 + 0.5) * view_scale
        gx, gy = np.meshgrid(c, c)
        self._grid_right = gx            # columns -> right of car
        self._grid_fwd = -gy + OBS_SIZE * view_scale * 0.25  # rows (top = ahead); car sits lower in image
        self.reset()

    # ------------------------------------------------------------------ API
    def new_track(self, difficulty=None, seed=None):
        if difficulty is not None:
            self.difficulty = difficulty
        self.seed = seed
        self.track = Track(road_width=self.road_width, difficulty=self.difficulty, seed=seed)
        return self.reset()

    def reset(self):
        p, h = self.track.start_pose()
        n = self.n_cars
        self.pos = np.tile(p, (n, 1)).astype(np.float64)
        self.heading = np.full(n, h)
        self.speed = np.zeros(n)
        self.alive = np.ones(n, dtype=bool)
        self.idx = np.zeros(n, dtype=np.int64)       # nearest centerline index
        self.progress = np.zeros(n, dtype=np.int64)  # total centerline points advanced (signed)
        self.best_progress = np.zeros(n, dtype=np.int64)
        self.last_improve = np.zeros(n, dtype=np.int64)
        self.steps = 0
        self.crash_reason = np.array([""] * n, dtype=object)
        return self._observe()

    def step(self, actions):
        actions = np.clip(np.asarray(actions, dtype=np.float64).reshape(self.n_cars, 2), -1, 1)
        steer, throttle = actions[:, 0], actions[:, 1]
        a = self.alive

        self.speed[a] += throttle[a] * self.accel
        self.speed[a] *= 1 - self.drag
        self.speed[a] = np.clip(self.speed[a], 0.0, self.max_speed)
        self.heading[a] += steer[a] * self.speed[a] / self.min_turn_radius
        self.pos[a, 0] += np.cos(self.heading[a]) * self.speed[a]
        self.pos[a, 1] += np.sin(self.heading[a]) * self.speed[a]

        prev_progress = self.progress.copy()
        self._update_progress()
        self.steps += 1

        # crashes / termination
        x = self.pos[:, 0].astype(int)
        y = self.pos[:, 1].astype(int)
        inside = (x >= 0) & (x < self.track.width) & (y >= 0) & (y < self.track.height)
        on_road = np.zeros(self.n_cars, dtype=bool)
        on_road[inside] = self.track.on_road[y[inside], x[inside]]
        crashed = a & ~on_road
        self.crash_reason[crashed] = "off_road"

        improved = self.progress > self.best_progress
        self.best_progress = np.maximum(self.best_progress, self.progress)
        self.last_improve[improved] = self.steps
        stalled = a & (self.steps - self.last_improve > self.stall_steps)
        self.crash_reason[stalled & ~crashed] = "stalled"
        self.alive &= ~(crashed | stalled)
        if self.steps >= self.max_steps:
            self.crash_reason[self.alive] = "timeout"
            self.alive[:] = False

        rewards = (self.progress - prev_progress) / self.track.n
        rewards[crashed] -= 0.1
        obs = self._observe()
        info = {"laps": self.fitness, "steps": self.steps, "reason": self.crash_reason.copy()}
        if self.render_mode == "human":
            self.render()
        return obs, rewards, ~self.alive, info

    @property
    def all_done(self):
        return not self.alive.any()

    @property
    def fitness(self):
        """Best progress along the track, in laps (1.0 = one full lap)."""
        return self.best_progress / self.track.n

    def get_image(self, i):
        """128x128x3 uint8 car-centric image for car i."""
        return self._render_views(np.array([i]))[0]

    # ------------------------------------------------------------ internals
    def _update_progress(self):
        c = self.track.center
        n = self.track.n
        window = int(self.max_speed / self.track.spacing) + 4
        offs = np.arange(-window, window + 1)
        cand = (self.idx[:, None] + offs[None, :]) % n           # (cars, W)
        d = np.linalg.norm(c[cand] - self.pos[:, None, :], axis=2)
        best = cand[np.arange(self.n_cars), d.argmin(1)]
        delta = (best - self.idx + n // 2) % n - n // 2
        self.progress[self.alive] += delta[self.alive]
        self.idx[self.alive] = best[self.alive]

    def _cast_rays(self):
        steps = np.linspace(0, self.ray_length, 64)
        ang = self.heading[:, None] + self.ray_angles[None, :]                    # (C, R)
        xs = self.pos[:, 0, None, None] + np.cos(ang)[..., None] * steps          # (C, R, S)
        ys = self.pos[:, 1, None, None] + np.sin(ang)[..., None] * steps
        xi = np.clip(xs.astype(int), 0, self.track.width - 1)
        yi = np.clip(ys.astype(int), 0, self.track.height - 1)
        road = self.track.on_road[yi, xi]
        road &= (xs >= 0) & (xs < self.track.width) & (ys >= 0) & (ys < self.track.height)
        off = ~road
        first = np.where(off.any(2), off.argmax(2), len(steps) - 1)
        return steps[first] / self.ray_length

    def _render_views(self, ids):
        h = self.heading[ids][:, None, None]
        fx, fy = np.cos(h), np.sin(h)        # forward vector
        rx, ry = -fy, fx                     # right vector (screen y points down)
        wx = self.pos[ids, 0, None, None] + self._grid_fwd * fx + self._grid_right * rx
        wy = self.pos[ids, 1, None, None] + self._grid_fwd * fy + self._grid_right * ry
        xi = wx.astype(np.int32)
        yi = wy.astype(np.int32)
        outside = (xi < 0) | (xi >= self.track.width) | (yi < 0) | (yi >= self.track.height)
        img = self.track.rgb[np.clip(yi, 0, self.track.height - 1), np.clip(xi, 0, self.track.width - 1)]
        img[outside] = GRASS
        # draw the car itself in the image
        cy = OBS_SIZE // 2 + int(OBS_SIZE * 0.25)
        cl, cw = int(20 / self.view_scale) or 1, int(10 / self.view_scale) or 1
        img[:, cy - cl // 2:cy + cl // 2, OBS_SIZE // 2 - cw // 2:OBS_SIZE // 2 + cw // 2] = (220, 30, 30)
        return img

    def _observe(self):
        obs = {
            "rays": self._cast_rays(),
            "speed": self.speed / self.max_speed,
        }
        if self.compute_images:
            obs["image"] = self._render_views(np.arange(self.n_cars))
        return obs

    # -------------------------------------------------------------- render
    def _make_sprites(self):
        def car(color):
            s = pygame.Surface((24, 12), pygame.SRCALPHA)
            pygame.draw.rect(s, color, (0, 0, 24, 12), border_radius=3)
            pygame.draw.rect(s, (30, 30, 40), (14, 2, 6, 8), border_radius=2)  # windshield
            return s
        self._car_sprites = {"alive": car((220, 30, 30)), "best": car((30, 120, 255)),
                             "dead": car((120, 60, 60, 110))}

    def render(self, show_rays=True, show_view=True):
        if self.screen is None:
            pygame.display.set_caption("CarEnv")
            self.screen = pygame.display.set_mode((self.track.width, self.track.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("monospace", 16)
            self._make_sprites()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.close()
                raise SystemExit

        self.screen.blit(self.track.surface, (0, 0))
        best = int(np.argmax(np.where(self.alive, self.best_progress, -1))) if self.alive.any() \
            else int(np.argmax(self.best_progress))
        order = sorted(range(self.n_cars), key=lambda i: (self.alive[i], i == best))
        for i in order:
            key = "dead" if not self.alive[i] else ("best" if i == best else "alive")
            spr = pygame.transform.rotate(self._car_sprites[key], -math.degrees(self.heading[i]))
            self.screen.blit(spr, spr.get_rect(center=(self.pos[i, 0], self.pos[i, 1])))

        if show_rays and self.alive[best]:
            rays = self._cast_rays()[best] * self.ray_length
            for ang, d in zip(self.ray_angles, rays):
                a = self.heading[best] + ang
                end = self.pos[best] + d * np.array([math.cos(a), math.sin(a)])
                pygame.draw.line(self.screen, (255, 255, 0), self.pos[best], end, 1)

        if show_view:
            img = self.get_image(best)
            view = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
            self.screen.blit(view, (self.track.width - OBS_SIZE - 10, 10))
            pygame.draw.rect(self.screen, (255, 255, 255),
                             (self.track.width - OBS_SIZE - 11, 9, OBS_SIZE + 2, OBS_SIZE + 2), 1)

        hud = [f"step {self.steps}", f"alive {self.alive.sum()}/{self.n_cars}",
               f"best laps {self.fitness.max():.3f}", f"speed {self.speed[best]:.2f}"]
        for k, t in enumerate(hud):
            self.screen.blit(self.font.render(t, True, (255, 255, 255)), (10, 10 + 18 * k))
        pygame.display.flip()
        if self.fps:
            self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
        self.screen = None


# ---------------------------------------------------------------- examples
def ray_controller(obs):
    """Simple hand-made policy: steer toward the side with more free space."""
    rays = obs["rays"]
    n = rays.shape[1]
    left, right = rays[:, : n // 2].mean(1), rays[:, n // 2 + 1:].mean(1)
    front = rays[:, n // 2]
    steer = np.clip((right - left) * 4, -1, 1)
    throttle = np.where(front > 0.5, 1.0, np.where(obs["speed"] > 0.4, -0.5, 0.3))
    return np.stack([steer, throttle], 1)


def keyboard_controller():
    k = pygame.key.get_pressed()
    steer = (k[pygame.K_RIGHT] or k[pygame.K_d]) - (k[pygame.K_LEFT] or k[pygame.K_a])
    throttle = (k[pygame.K_UP] or k[pygame.K_w]) - (k[pygame.K_DOWN] or k[pygame.K_s])
    return np.array([[steer, throttle]], dtype=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="many cars with a ray-based controller")
    ap.add_argument("--difficulty", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cars", type=int, default=30)
    args = ap.parse_args()

    n = args.cars if args.demo else 1
    env = CarEnv(n_cars=n, difficulty=args.difficulty, seed=args.seed, render_mode="human",
                 max_steps=10**9 if not args.demo else 3000, stall_steps=10**9 if not args.demo else 150)
    obs = env.reset()
    if args.demo:
        noise = np.random.default_rng(0).normal(0, 0.3, (n, 2))
    while True:
        if args.demo:
            act = ray_controller(obs) + noise
        else:
            act = keyboard_controller()
        obs, rew, done, info = env.step(act)
        if env.all_done:
            print("episode done, fitness (laps):", np.round(env.fitness, 3))
            obs = env.reset()
