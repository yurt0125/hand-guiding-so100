import cv2
import os

def main():
    # ===== 你可以改这几个参数 =====
    squares_x = 6
    squares_y = 8
    square_length_m = 0.035   # 3.5 cm
    marker_length_m = 0.020   # 2.0 cm

    dpi = 300
    paper_w_mm = 210   # A4 width
    paper_h_mm = 297   # A4 height

    out_path = "charuco_board.png"

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_m,
        marker_length_m,
        dictionary
    )

    px_per_mm = dpi / 25.4
    img_w = int(paper_w_mm * px_per_mm)
    img_h = int(paper_h_mm * px_per_mm)

    board_img = board.generateImage((img_w, img_h), marginSize=40, borderBits=1)
    cv2.imwrite(out_path, board_img)

    print(f"Saved: {os.path.abspath(out_path)}")
    print("Print notes:")
    print("1. Disable scaling in printer dialog")
    print("2. Print at 100% size")
    print("3. Paste on a rigid flat board")
    print("4. Measure actual square size after printing")

if __name__ == "__main__":
    main()
