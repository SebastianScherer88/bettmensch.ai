from typing import Literal, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Colouring
LOGO_IMAGE = cv2.imread("./image/logo_transparent.png")


# --- ArgoWorkflows dashboard embedding config
class MlflowDisplaySettings(BaseSettings):
    width: int = 1200
    height: int = 1000
    scrolling: bool = True
    host: str = "mlflow.svc.cluster.local:5000"
    starting_endpoint: str = "#/models"

    model_config = SettingsConfigDict(env_prefix="mlflow_backend_")


# --- Mlflow dashboard embedding config
class ArgoWorkflowsDisplaySettings(BaseSettings):
    width: int = 1200
    height: int = 1000
    scrolling: bool = True
    host: str = "argo.svc.cluster.local:2746"
    starting_endpoint: str = "workflows?namespace=argo&limit=50"

    model_config = SettingsConfigDict(env_prefix="argo_workflows_backend_")


class CustomTheme(BaseModel):
    """Defaults to "dark" if not specified. Colors MUST be HEX values, e.g.

    "light" theme:
    primaryColor="#FF4B4B"
    backgroundColor="#FFFFFF"
    secondaryBackgroundColor="#F0F2F6"
    textColor="#31333F"
    """

    primaryColor: str = "#FF4B4B"
    backgroundColor: str = "#0E1117"  # main panel
    secondaryBackgroundColor: str = "#262730"  # sidebar panel
    textColor: str = "#FAFAFA"


class DarkTheme(CustomTheme):
    primaryColor: str = "#FF4B4B"
    backgroundColor: str = "#0E1117"  # main panel
    secondaryBackgroundColor: str = "#262730"  # sidebar panel
    textColor: str = "#FAFAFA"


class LightTheme(CustomTheme):
    primaryColor: str = "#FF4B4B"
    backgroundColor: str = "#FFFFFF"  # main panel
    secondaryBackgroundColor: str = "#F0F2F6"  # sidebar panel
    textColor: str = "#31333F"


def hex_to_rgb(hex: str):
    h = hex.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # noqa: E203


def hex_to_bgr(hex: str):
    rgb = hex_to_rgb(hex)
    return (rgb[2], rgb[1], rgb[0])


def hex_to_channel(hex: str, channel: Literal["RGB", "BGR"]):
    if channel == "RGB":
        return hex_to_rgb(hex)
    elif channel == "BGR":
        return hex_to_bgr(hex)
    else:
        raise ValueError(
            f"Channel argument {channel} is invalid. Supported channels: "
            "RGB,BGR"
        )


def get_colors(
    theme: Literal["dark", "light", "custom"] = "dark"
) -> CustomTheme:
    if theme == "dark":
        colors = DarkTheme()
    elif theme == "light":
        colors = LightTheme()
    else:
        colors = CustomTheme(
            primaryColor=st.get_option("theme.primaryColor"),
            backgroundColor=st.get_option("theme.backgroundColor"),
            secondaryBackgroundColor=st.get_option(
                "theme.secondaryBackgroundColor"
            ),
            textColor=st.get_option("theme.textColor"),
        )

    return colors


def color_background(
    image,
    new_background_color: Optional[Tuple[int, int, int]] = hex_to_bgr(
        DarkTheme().backgroundColor
    ),
):
    """Takes an image array of shape (w,h,3) and replaces all white pixels with
    the new_background_color."""

    background_mask = np.expand_dims((image == 255).all(-1).astype(int), -1)
    image_no_background = image - 255 * background_mask

    new_background = np.stack(
        [
            background_mask[:, :, 0] * new_background_color[0],
            background_mask[:, :, 0] * new_background_color[1],
            background_mask[:, :, 0] * new_background_color[2],
        ],
        -1,
    )

    image_new_background = image_no_background + new_background

    return image_new_background


def add_logo(sidebar: bool = False):
    colors = get_colors("custom")
    if colors.primaryColor is None:
        colors = get_colors("dark")

    if sidebar:
        background_color = colors.secondaryBackgroundColor
    else:
        background_color = colors.backgroundColor

    st.image(
        color_background(LOGO_IMAGE, hex_to_channel(background_color, "BGR")),
        channels="BGR",
    )
