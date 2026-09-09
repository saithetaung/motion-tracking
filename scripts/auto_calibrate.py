"""
One-time, zero-typing calibration: hold a printed checkerboard (9x6 inner
corners, any size) in front of the webcam, move it around for ~10 seconds.
OpenCV auto-detects corners and solves for focal length. No manual point
picking, no distance measuring.

Run once per camera: python scripts/auto_calibrate.py
"""
import cv2
import numpy as np
import yaml
import os

BOARD_SIZE = (9, 6)
SQUARE_SIZE_M = 0.025  # only affects absolute scale of intrinsics matrix, not focal_length_px validity
TARGET_SAMPLES = 15


def main():
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE_M

    objpoints, imgpoints = [], []
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print(f"Show a {BOARD_SIZE[0]}x{BOARD_SIZE[1]} checkerboard to the camera. "
          f"Collecting {TARGET_SAMPLES} samples automatically...")

    gray = None
    while len(objpoints) < TARGET_SAMPLES:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, BOARD_SIZE, corners, found)
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"  captured {len(objpoints)}/{TARGET_SAMPLES}")

        cv2.putText(display, f"Samples: {len(objpoints)}/{TARGET_SAMPLES}",
                     (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Calibration - move the board around", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print("Not enough samples captured — try again with better lighting.")
        return

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    focal_length_px = float(camera_matrix[0, 0])

    os.makedirs("configs", exist_ok=True)
    with open("configs/camera_webcam.yaml", "w") as f:
        yaml.safe_dump({"focal_length_px": focal_length_px}, f)

    print(f"\nDone. focal_length_px = {focal_length_px:.1f}")
    print("Saved to configs/camera_webcam.yaml — run_demo.py will pick it up automatically.")


if __name__ == "__main__":
    main()