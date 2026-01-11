import pygame
import random
from enum import Enum

GRID_SIZE = 16
CELL_SIZE = 32
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

SNAKE_COLOR = (80, 200, 80)
FOOD_COLOR = (200, 50, 50)
BG_COLOR = (20, 20, 20)
GRID_COLOR = (40, 40, 40)
TEXT_COLOR = (250, 250, 250)


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class SnakeGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake 16x16")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 24)
        self.reset()

    def reset(self):
        # Start snake in the middle
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = Direction.RIGHT
        self.pending_direction = self.direction
        self.score = 0
        self.game_over = False
        self._place_food()

    def _place_food(self):
        # Choose a random empty cell for the apple
        all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available = list(set(all_cells) - set(self.snake))
        self.food = random.choice(available) if available else None

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if not self.game_over:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        self._set_direction(Direction.UP)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        self._set_direction(Direction.DOWN)
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        self._set_direction(Direction.LEFT)
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        self._set_direction(Direction.RIGHT)
                else:
                    # Any key after game over restarts the game
                    self.reset()
        return True

    def _set_direction(self, new_dir: Direction):
        # Prevent 180° turns directly into yourself
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        if new_dir != opposite[self.direction]:
            self.pending_direction = new_dir

    def update(self):
        if self.game_over:
            return

        # Apply the latest valid direction
        self.direction = self.pending_direction

        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)

        # Wall collision
        if not (0 <= new_head[0] < GRID_SIZE and 0 <= new_head[1] < GRID_SIZE):
            self.game_over = True
            return

        # Self collision
        if new_head in self.snake:
            self.game_over = True
            return

        # Move snake
        self.snake.insert(0, new_head)

        # Apple eaten
        if self.food is not None and new_head == self.food:
            self.score += 1
            self._place_food()
        else:
            # Just move forward (remove tail)
            self.snake.pop()

    def draw_grid(self):
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WIDTH, y))

    def draw(self):
        self.screen.fill(BG_COLOR)
        self.draw_grid()

        # Draw apple
        if self.food is not None:
            fx, fy = self.food
            pygame.draw.rect(
                self.screen,
                FOOD_COLOR,
                (fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE),
            )

        # Draw snake
        for (sx, sy) in self.snake:
            rect = pygame.Rect(sx * CELL_SIZE, sy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, SNAKE_COLOR, rect)

        # Draw score
        score_surf = self.font.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.screen.blit(score_surf, (10, 10))

        # Game over text
        if self.game_over:
            msg = "Game Over - Press any key"
            msg_surf = self.font.render(msg, True, TEXT_COLOR)
            msg_rect = msg_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(10)  # game speed (FPS)

        pygame.quit()


if __name__ == "__main__":
    SnakeGame().run()