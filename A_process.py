import subprocess
import sys
import schedule
import time

def run_pipeline():
    try:
        subprocess.run([sys.executable, 'fetch.py', "-t", "0050.TW", "-y", "10"], check=True)
        subprocess.run([sys.executable, 'features.py'], check=True)
        subprocess.run([sys.executable, 'rolling.py'], check=True)
        subprocess.run([sys.executable, 'pretidy.py'], check=True)
        subprocess.run([sys.executable, 'bayesian_unified.py', "--window-weeks", "10", "--bins", "4"], check=True)
        subprocess.run([sys.executable, 'backtest_signal_based.py'], check=True)
        subprocess.run([sys.executable, 'send_discord_webhook.py'], check=True)
        print("Pipeline completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the pipeline: {e}")


if __name__ == "__main__":
    run_pipeline()
    print("Initial run completed. Setting up scheduler...")
    # 每天20點執行
    schedule.every().day.at("20:00").do(run_pipeline)

    print("Scheduler started. Waiting for the next run...")
    while True:
        schedule.run_pending()
        time.sleep(30)
