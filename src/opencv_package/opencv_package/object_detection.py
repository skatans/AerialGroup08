
def main():
    frame = cv2.imread("/home/tuisku/Downloads/portti.jpeg", cv2.IMREAD_COLOR)
    #cv2.imshow("Received frame", frame)
    cv2.waitKey(1)
    # Convert the image to grayscale for shape detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blur = cv2.GaussianBlur(gray,(15,15),0)
    #th3 = cv2.medianBlur(gray,5)
    #th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    '''
    # Rectangular Kernel
    kernel_rq = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # Elliptical Kernel
    kernel_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    sharpen = cv2.filter2D(blur, -1, kernel_rq)
    '''

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame = blur

    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check the number of vertices in the approximated contour
        vertices = len(approx)

        if vertices == 4:
            shape = "Square/Rectangle"
        elif vertices > 4:
            shape = "Circle"
        else:
            shape = "Other"

        # Draw the contour and label the shape
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with detected shapes
    cv2.imshow("Shape Detection", frame)


if __name__ == '__main__':
    main()
