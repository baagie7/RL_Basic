from gym.envs.registration import register

register(
    id='GridWorld-v0',
    entry_point='lib.envs.grid_world:GridWorld'
)

register(
    id='WindyGridWorld-v0',
    entry_point='lib.envs.windy_grid_world:WindyGridworld'
)
