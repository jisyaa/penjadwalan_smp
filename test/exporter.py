import os, json, datetime

def save_generation(run_dir, gen, fitness, schedule):
    path = f"{run_dir}/gen_{gen:04d}.json"
    with open(path, "w") as f:
        json.dump({
            "generation": gen,
            "fitness": fitness,
            "schedule": schedule
        }, f, indent=2)
