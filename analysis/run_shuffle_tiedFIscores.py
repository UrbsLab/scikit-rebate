"""
Submits SLURM/LSF jobs to randomly shuffle tied feature importance scores involving predictive features in Results.txt files.  
"""
import argparse
import os
import time

LSF_TEMPLATE = """#!/bin/bash
#BSUB -J {job_name}
#BSUB -o logs/{job_name}.out
#BSUB -e logs/{job_name}.err
#BSUB -n 1
#BSUB -R "rusage[mem=4096]"
#BSUB -W 01:00
#BSUB -q i2c2_normal

python job_shuffle_tiedFIscores.py --input_file "{input_file}"
"""

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.out
#SBATCH --error=logs/{job_name}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --partition=defq

python job_shuffle_tiedFIscores.py --input_file "{input_file}"
"""

def find_results_files(dir_path, algorithm_names):
    results_files = []
    # Recursively search for _Results.txt files
    for root, dirs, files in os.walk(dir_path):
        # Name of the directory containing the files
        parent_dir = os.path.basename(root)
        grandparent_dir = os.path.basename(os.path.dirname(root))
    
        # Ensure algorithm directories are directly inside a "Results" directory
        if grandparent_dir != 'Results':
            continue
    
        # Apply algorithm restriction, if specified
        if algorithm_names is not None and parent_dir.lower() not in algorithm_names:
            continue
    
        for file in files:
            if not file.endswith('_Results.txt'):
                continue
    
            results_path = os.path.join(root, file)
            results_files.append(results_path)

    return results_files

def generate_hpc_jobs(data_dir, results_files, job_dir, suffix, hpctype="slurm"):
    os.makedirs(job_dir, exist_ok=True)
    if hpctype == "lsf":
        TEMPLATE = LSF_TEMPLATE
    elif hpctype == "slurm":
        TEMPLATE = SLURM_TEMPLATE
    else:
        raise Exception(f"Unsupported HPC type: {hpctype}")
    job_paths = []
    for result_file in results_files:
        rel_path = os.path.relpath(result_file, data_dir)
        parts = rel_path.split(os.sep) # Split into parts
        last_dirs_and_file = parts[-3:]  # [-3:] = last 2 dirs + file
        safe_path = "_".join(last_dirs_and_file) # Join with underscore
        safe_path = safe_path.replace(".", "")
        base_name = os.path.splitext(safe_path)[0]
        
        job_name = f"{base_name}_{suffix}"
        job_path = os.path.join(job_dir, f"{job_name}.sh")
        with open(job_path, "w") as f:
            f.write(TEMPLATE.format(
                job_name=job_name,
                input_file=result_file,
            ))
        job_paths.append(job_path)
    return job_paths

def main():
    parser = argparse.ArgumentParser(description="Generate and submit jobs to randomly shuffle tied feature importance scores involving predictive features in Results.txt files.")
    parser.add_argument('--basedir', required=True, help='Base directory to search for Results.txt files.')
    parser.add_argument('--hpctype', choices=['lsf', 'slurm'], default='slurm', help='Scheduler type: lsf or slurm.')
    parser.add_argument('--script', default='job_shuffle_tiedFIscores.py', help='Python script to run.')
    parser.add_argument('--jobdir', default='jobs', help='Directory to store job scripts.')
    parser.add_argument('--logdir', default='logs', help='Directory to store output and error logs.')
    parser.add_argument('--algorithms', nargs='+', default='all', help='Algorithm subdirectories to search through in Results directories.')

    args = parser.parse_args()

    os.makedirs(args.jobdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    # job_count = 0
    dir_path = args.basedir  # only look inside basedir

    # Determine algorithm restrictions
    if args.algorithms == 'all':
        algorithm_names = None
    else: # restrict search to only the specified algorithms
        algorithm_names = {algorithm.lower() for algorithm in args.algorithms}

    suffix = time.strftime("%Y%m%d_%H%M%S")
    # Find "_Results.txt" files to operate on (i.e. submit a job for and inspect for ties)
    results_files = find_results_files(dir_path, algorithm_names)
    job_files = generate_hpc_jobs(dir_path, results_files, args.jobdir, suffix, hpctype=args.hpctype)

    print(f"Generated {len(job_files)} job scripts in: {args.jobdir}")
    
    # Automatically submit jobs
    for job_file in job_files:
        if args.hpctype == "lsf":
            os.system(f"bsub < {job_file}")
        elif args.hpctype == "slurm":
            os.system(f"sbatch {job_file}")
        else:
            raise Exception(f"Unsupported HPC type: {args.hpctype}")
        print(f"Submitted job: {job_file}")
    
    print("All jobs submitted successfully.")

if __name__ == "__main__":
    main()