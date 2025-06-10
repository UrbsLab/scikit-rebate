import os
import time

# ALGORITHMS = ["mutual_info", "relieff10", "relieff100", "surf", "surfstar", "multisurf", "multisurfstar", "swrfstar"]

ALGORITHMS = ["mutual_info", "surfstar", "swrfstar", "multisurf", "multisurfstar"]

LSF_TEMPLATE = """
#!/bin/bash
#BSUB -J {job_name}
#BSUB -o logs/{job_name}.out
#BSUB -e logs/{job_name}.err
#BSUB -n 1
#BSUB -R "rusage[mem=4096]"
#BSUB -W 01:00
#BSUB -q i2c2_normal

python run_feature_selection.py --algorithm {algorithm} --input_file "{input_file}" --output_dir "{output_dir}"
"""

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_dataset_files(root_dir):
    dataset_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "EDM" in os.path.basename(dirpath):
            for filename in filenames:
                if filename.endswith(".txt"):
                    dataset_files.append(os.path.join(dirpath, filename))
    return dataset_files

def generate_lsf_jobs(dataset_files, algorithms, job_dir, output_dir, suffix):
    ensure_dir(job_dir)
    ensure_dir(os.path.join(job_dir, "logs"))
    job_paths = []
    for dataset in dataset_files:
        base_name = os.path.splitext(os.path.basename(dataset))[0]
        for algo in algorithms:
            job_name = f"{base_name}_{algo}_{suffix}"
            job_path = os.path.join(job_dir, f"{job_name}.sh")
            with open(job_path, "w") as f:
                f.write(LSF_TEMPLATE.format(
                    job_name=job_name,
                    algorithm=algo,
                    input_file=dataset,
                    output_dir=output_dir
                ))
            job_paths.append(job_path)
    return job_paths

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root directory to search for dataset files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output results")
    parser.add_argument("--job_dir", default="lsf_jobs", help="Directory to save LSF job scripts")
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS, help="Algorithms to run")
    args = parser.parse_args()

    # if not args.algorithms:
    #     args.algorithms = ALGORITHMS

    suffix = time.strftime("%Y%m%d_%H%M%S")
    dataset_files = find_dataset_files(args.data_root)
    job_files = generate_lsf_jobs(dataset_files, args.algorithms, args.job_dir, args.output_dir, suffix)

    print(f"Generated {len(job_files)} LSF job scripts in: {args.job_dir}")

    # Automatically submit jobs
    for job_file in job_files:
        # os.system(f"bsub < {job_file}")
        print(f"Would submit job: {job_file}")

if __name__ == "__main__":
    main()
