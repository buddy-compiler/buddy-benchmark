#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import getpass
from pathlib import Path

def execute_remote_benchmark(host, username, program_path, output_file):
    try:
        program_path = Path(program_path).resolve()
        if not program_path.exists():
            raise FileNotFoundError(f"the executable file does not exist: {program_path}")
        program_name = program_path.name
        remote_dir = f"/tmp/bench_{os.urandom(4).hex()}"
        remote_program_path = f"{remote_dir}/{program_name}"
        create_dir_cmd = f"ssh {username}@{host} 'mkdir -p {remote_dir}'"
        subprocess.run(create_dir_cmd, shell=True, check=True, capture_output=True)
        scp_cmd = f"scp {program_path} {username}@{host}:{remote_program_path}"
        subprocess.run(scp_cmd, shell=True, check=True, capture_output=True)
        exec_cmd = (f"ssh {username}@{host} "
                    f"'chmod +x {remote_program_path} && "
                    f"cd {remote_dir} && ./{program_name} "
                    f"--benchmark_out=result.json --benchmark_out_format=json'")
        subprocess.run(exec_cmd, shell=True, check=True, capture_output=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        retrieve_cmd = f"scp {username}@{host}:{remote_dir}/result.json {output_file}"
        subprocess.run(retrieve_cmd, shell=True, check=True, capture_output=True)
        cleanup_cmd = f"ssh {username}@{host} 'rm -rf {remote_dir}'"
        subprocess.run(cleanup_cmd, shell=True, check=True, capture_output=True)

        print(f"\n succeeded: {output_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f" Command failed: {e.cmd}")
        if e.stderr:
            print(f"Failed: {e.stderr.decode().strip()}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='execute a benchmark program on a remote server via SSH and retrieve the JSON results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--host', required=True, help='server IP')
    parser.add_argument('--user', default=getpass.getuser(), help='SSH username')
    parser.add_argument('--program', required=True, help='local program path')
    parser.add_argument('--output', help='local output file path')

    args = parser.parse_args()
    program_path = Path(args.program).resolve()
    if args.output:
        output_file = Path(args.output).resolve()
    else:
        parent_dir = program_path.parent
        if parent_dir == Path.cwd():
            print(" Failed: please set --output <dir>/<file.json>")
            sys.exit(1)
        dir_name = parent_dir.name
        output_dir = Path.cwd() / "output"
        output_file = output_dir / f"{dir_name}.json"
        print(f"output_file: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Output: {output_file}")

    success = execute_remote_benchmark(
        host=args.host,
        username=args.user,
        program_path=program_path,
        output_file=output_file
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()