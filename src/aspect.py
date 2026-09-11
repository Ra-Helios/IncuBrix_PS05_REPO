"""Supported output aspect ratios.

Parsing and resolution helpers so the pipeline can validate a requested
W:H ratio against ``config.aspect.supported_ratios`` and derive a matching
output pixel resolution.
"""

from typing import Tuple


def parse_ratio(text: str) -> Tuple[int, int]:
    """Parse a 'W:H' string into a (width, height) pair of positive ints.

    Args:
        text: Aspect ratio text such as "9:16" or "1:1".

    Returns:
        Tuple of (width, height) components.

    Raises:
        ValueError: If ``text`` is not a valid positive 'W:H' pair.
    """
    try:
        w_part, h_part = text.split(":")
        width = int(w_part)
        height = int(h_part)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid aspect ratio {text!r} (expected 'W:H', e.g. '9:16')"
        ) from exc

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid aspect ratio {text!r} (expected positive width:height)"
        )
    return width, height


def resolve_output_dimensions(output_width: int, width: int, height: int) -> Tuple[int, int]:
    """Derive output pixel dimensions that preserve the target W:H ratio.

    The output width is kept; the height is derived so the rendered video
    has exactly the requested aspect ratio.

    Args:
        output_width: Desired output width in pixels.
        width: Aspect ratio width component.
        height: Aspect ratio height component.

    Returns:
        (output_width, output_height) matching the W:H ratio.
    """
    output_height = round(output_width * height / width)
    return output_width, output_height