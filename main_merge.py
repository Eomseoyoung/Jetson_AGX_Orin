from multiprocessing import Process
import Backcam_merge as Backcam
import Frontcam_merge as Frontcam

def main():
    backcam_process = Process(target=Backcam.run)
    frontcam_process = Process(target=Frontcam.run)

    backcam_process.start()
    frontcam_process.start()

    backcam_process.join()
    frontcam_process.join()

if __name__ == "__main__":
    main()
