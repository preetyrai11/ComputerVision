import cv2
import numpy as np

def detect_rectangles(image):
    """
    Detect rectangles in the image using contour detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter out small contours
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:  # Quadrilateral
                rectangles.append(approx)
    return rectangles

def calculate_iou(rect1, rect2):
    """
    Calculate Intersection over Union (IoU) between two rectangles.
    """
    rect1_area = cv2.contourArea(rect1)
    rect2_area = cv2.contourArea(rect2)
    intersection = cv2.bitwise_and(rect1, rect2)
    intersection_area = cv2.contourArea(intersection)
    iou = intersection_area / float(rect1_area + rect2_area - intersection_area)
    return iou

def correct_perspective(rectangle, image):
    """
    Correct the perspective of a rectangle to a top-down view.
    """
    (x, y, w, h) = cv2.boundingRect(rectangle)
    src_pts = rectangle.reshape(-1, 2).astype(np.float32)
    dst_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected_image = cv2.warpPerspective(image, M, (w, h))
    return corrected_image

def main():
    image = cv2.imread('rectanglesOverlap.png')
    rectangles = detect_rectangles(image)
    if len(rectangles) != 2:
        print("Exactly two rectangles expected.")
        return
    iou = calculate_iou(rectangles[0], rectangles[1])
    print("IoU:", iou)
    corrected_images = [correct_perspective(rect, image) for rect in rectangles]
    cv2.imwrite('corrected_rect1.jpg', corrected_images[0])
    cv2.imwrite('corrected_rect2.jpg', corrected_images[1])

if __name__ == '__main__':
    main()
    

