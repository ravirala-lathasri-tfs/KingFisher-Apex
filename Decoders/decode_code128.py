import cv2
from pyzbar.pyzbar import decode, ZBarSymbol


def try_decode(image, method):
    results = decode(
        image,
        symbols=[ZBarSymbol.CODE128]
    )

    if results:
        for barcode in results:
            value = barcode.data.decode(
                "utf-8",
                errors="replace"
            )

            print()
            print("SUCCESS")
            print("Method :", method)
            print("Type   :", barcode.type)
            print("Value  :", value)

            return value

    return None


def rotate_image(image, angle):
    h, w = image.shape[:2]

    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        borderValue=255
    )


def decode_code128(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("ERROR: Could not read image.")
        return None

    print("Image size:", image.shape)

    # ------------------------------------------------
    # 1. Original image
    # ------------------------------------------------
    result = try_decode(image, "original")

    if result:
        return result

    # ------------------------------------------------
    # 2. Grayscale
    # ------------------------------------------------
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    result = try_decode(gray, "grayscale")

    if result:
        return result

    # ------------------------------------------------
    # 3. Enlarged grayscale
    # ------------------------------------------------
    enlarged = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    result = try_decode(
        enlarged,
        "2x grayscale"
    )

    if result:
        return result

    # ------------------------------------------------
    # 4. Otsu threshold
    # ------------------------------------------------
    _, otsu = cv2.threshold(
        enlarged,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    result = try_decode(
        otsu,
        "Otsu"
    )

    if result:
        return result

    # ------------------------------------------------
    # 5. Adaptive threshold
    # ------------------------------------------------
    adaptive = cv2.adaptiveThreshold(
        enlarged,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    result = try_decode(
        adaptive,
        "adaptive threshold"
    )

    if result:
        return result

    # ------------------------------------------------
    # 6. Try rotations on the ORIGINAL grayscale
    # ------------------------------------------------
    for angle in range(-15, 16):

        rotated = rotate_image(
            gray,
            angle
        )

        result = try_decode(
            rotated,
            f"rotation {angle} degrees"
        )

        if result:
            return result

    # ------------------------------------------------
    # 7. Try rotations on enlarged image
    # ------------------------------------------------
    for angle in range(-15, 16):

        rotated = rotate_image(
            enlarged,
            angle
        )

        result = try_decode(
            rotated,
            f"2x + rotation {angle} degrees"
        )

        if result:
            return result

    print()
    print("No valid Code 128 barcode decoded.")

    return None


if __name__ == "__main__":
    image_path = r"C:\Users\lathasri.ravirala\Downloads\test\Dataset\barcode (37).png"

    result = decode_code128(image_path)

    print()
    print("Final result:", result)
