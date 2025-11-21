from browsergym.register import REGISTERED_ENVS

print("\n=== All WebLINX tasks registered in BrowserGym ===")
weblinx_tasks = [
    env_id for env_id in sorted(REGISTERED_ENVS.keys())
    if env_id.startswith("weblinx.")
]

for task in weblinx_tasks:
    print(task)

print("\nTotal WebLINX tasks:", len(weblinx_tasks))
