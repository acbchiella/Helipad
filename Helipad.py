
from typing import Optional
import numpy as np
import numpy.typing as npt
import cv2
import matplotlib.pyplot as plt


class Helipad:

    def __init__(
            self,
            image: Optional[npt.ArrayLike] = None,
            color_mask: Optional[npt.ArrayLike] = None
    ) -> None:
        self._image = image
        self._color_mask = color_mask

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image: Optional[npt.ArrayLike] = None):
        self._image = image

    @property
    def color_mask(self) -> Optional[npt.ArrayLike]:
        return self._color_mask

    @color_mask.setter
    def color_mask(self, color_mask: Optional[npt.ArrayLike] = None):
        self._color_mask = color_mask

    @staticmethod
    def _apply_color_mask(image: npt.ArrayLike, color: npt.ArrayLike) -> npt.ArrayLike:
        assert color.shape == (3,)

        color_tolerance = 35
        lower_bound = np.array(
            [
                max(0, color[0] - color_tolerance),
                max(0, color[1] - color_tolerance),
                max(0, color[2] - color_tolerance)
            ]
        )
        upper_bound = np.array(
            [
                min(255, color[0] + color_tolerance),
                min(255, color[1] + color_tolerance),
                min(255, color[2] + color_tolerance)
            ]
        )
        mask = cv2.inRange(image, lower_bound, upper_bound)
        filtered_image = image & cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    def find_circle(self, filtered_image: Optional[npt.ArrayLike] = None) -> npt.ArrayLike:
        if filtered_image is None:
            filtered_image = self._image

        gray_blurred = cv2.GaussianBlur(filtered_image, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,             # Inverse ratio of accumulator resolution
            minDist=gray_blurred.shape[0],       # Minimum distance between detected centers
            param1=250,        # Upper threshold for the internal Canny edge detector
            param2=35,        # Threshold for center detection
            minRadius=0,      # Minimum radius of the circles
            maxRadius=0       # Maximum radius of the circles (0 means no maximum)
        )

        helipad_circle = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # choose the circle with greater radius
            helipad_circle = circles[0, :][0]
            for circle in circles[0, :]:
                if helipad_circle[2] < circle[2]:
                    helipad_circle = circle

        return helipad_circle

    def find_h(self) -> None:
        # TODO
        pass

    def find_helipad(self) -> npt.ArrayLike:
        filtered_image = Helipad._apply_color_mask(self._image, self._color_mask)
        circle = self.find_circle(filtered_image)

        # TODO
        h = self.find_h()

        return circle

    def draw_bounding_box():
        # TODO
        pass

    def draw_circle(
            self,
            circle: npt.ArrayLike,
            color: npt.ArrayLike = np.array([0, 255, 0])
    ):
        image = np.copy(self._image)

        if circle is None:
            return image

        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(image, (circle[0], circle[1]), circle[2], color, 10)
        cv2.circle(image, (circle[0], circle[1]), 6, color, -1)

        return image

    @staticmethod
    def show_image(image: npt.ArrayLike):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Image")
        plt.axis('off')
        plt.show()
