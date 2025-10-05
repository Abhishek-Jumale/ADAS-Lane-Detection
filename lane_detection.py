import cv2
import numpy as np


def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)

    polygon = np.array([[
        (0, height),
        (width // 2, int(height * 0.6)),
        (width, height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked



def draw_lane_poly(frame, binary_img):
    """
    Find lane pixels, fit quadratic curves (x = a*y^2 + b*y + c)
    and overlay the curved lane area on the frame.
    """
    ys, xs = np.nonzero(binary_img)

    if len(xs) < 100:
        return frame  

    mid_x = frame.shape[1] // 2
 
    left_x = xs[xs < mid_x]
    left_y = ys[xs < mid_x]
    right_x = xs[xs > mid_x]
    right_y = ys[xs > mid_x]

    left_fit = right_fit = None
    if len(left_x) > 0 and len(left_y) > 0:
        left_fit = np.polyfit(left_y, left_x, 2)
    if len(right_x) > 0 and len(right_y) > 0:
        right_fit = np.polyfit(right_y, right_x, 2)


    plot_y = np.linspace(0, frame.shape[0] - 1, frame.shape[0])

  
    if left_fit is not None:
        left_fitx = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
    else:
        left_fitx = np.array([])

    if right_fit is not None:
        right_fitx = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
    else:
        right_fitx = np.array([])

    lane_overlay = np.zeros_like(frame)
    if len(left_fitx) and len(right_fitx):
        pts_left = np.array([np.transpose(np.vstack([left_fitx, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, plot_y])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(lane_overlay, np.int32([pts]), (0, 255, 0))

   
    combo = cv2.addWeighted(frame, 1, lane_overlay, 0.3, 0)
    return combo



video_path = "11367262-hd_1920_1080_60fps.mp4"
cap = cv2.VideoCapture(video_path)


ret, test_frame = cap.read()
if not ret:
    print("Error: Could not read video file.")
    cap.release()
    exit()

height, width = test_frame.shape[:2]


out = cv2.VideoWriter('lane_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("🚗 Lane detection started... Press 'Q' to stop.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

   
    cropped_edges = region_of_interest(edges)

 
    lane_result = draw_lane_poly(frame, cropped_edges)

 
    out.write(lane_result)

    
    cv2.imshow("Edges", edges)
    cv2.imshow("ROI", cropped_edges)
    cv2.imshow("Lane Detection (Curved)", lane_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
print(" Lane detection completed. Output saved as 'lane_output.mp4'")
