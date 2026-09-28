def format_header(header: str, line: str, width: int) -> str:
    if width < len(header):
        raise ValueError("header text is wider than total header width")

    remaining = width - len(header) - 2  # Spaces left and right
    if remaining % 2 == 0:
        left_len = remaining // 2
        right_len = left_len
    else:
        left_len = (remaining + 1) // 2
        right_len = remaining - left_len

    return line * left_len + " " + header + " " + line * right_len


def print_header(header: str, line: str, width: int) -> None:
    print(format_header(header, line, width))
