import cv2
import numpy as np


def decode_qr(image):
    """
    Detect and decode QR code from an image.

    Parameters
    ----------
    image : numpy.ndarray
        OpenCV BGR image.

    Returns
    -------
    str or None
        Decoded QR/Barcode value.
    """

    if image is None:
        return None

    detector = cv2.QRCodeDetector()

    try:
        data, points, _ = detector.detectAndDecode(image)

        if data:
            return data.strip()

    except Exception:
        pass

    return None


def detect_qr_details(image):
    """
    Detect QR code and return decoded value + corner points.
    """

    if image is None:
        return None, None

    detector = cv2.QRCodeDetector()

    try:
        data, points, _ = detector.detectAndDecode(image)

        if data:
            return data.strip(), points

    except Exception:
        pass

    return None, None


def draw_detection(image, points):
    """
    Draw a bounding box around detected QR code.
    """

    if image is None or points is None:
        return image

    output = image.copy()

    points = points.astype(int)

    points = points.reshape(-1, 2)

    for i in range(len(points)):
        start = tuple(points[i])
        end = tuple(points[(i + 1) % len(points)])

        cv2.line(
            output,
            start,
            end,
            (0, 255, 0),
            3
        )

    return output


def decode_image_bytes(image_bytes):
    """
    Convert Streamlit camera/image bytes into OpenCV image
    and decode QR code.
    """

    if image_bytes is None:
        return None, None

    try:

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return None, None

        value, points = detect_qr_details(image)

        return value, image

    except Exception:
        return None, None