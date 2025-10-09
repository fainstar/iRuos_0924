import subprocess
import sys

def main():
    try:
        subprocess.run([sys.executable, 'fetch.py',"-t","00631L.TW","-y","10"])
        subprocess.run([sys.executable, 'features.py'])
        subprocess.run([sys.executable, 'rolling.py'])
        subprocess.run([sys.executable, 'pretidy.py'])
        subprocess.run([sys.executable, 'bayesian_unified.py',"--window-weeks","10","--bins","4"])
        subprocess.run([sys.executable, 'backtest_signal_based.py'])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the pipeline: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())