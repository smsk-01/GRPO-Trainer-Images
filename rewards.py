import re


def extract_action(response: str) -> str | None:
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'action':\s*'(?P<act>\w+)'"
    action_pattern_1 = r"'action':\s*(?P<act>\w+)"
    m = re.search(answer_tag_pattern, response, re.DOTALL)
    if not m:
        return None
    text = m.group(1)
    for pat in (action_pattern, action_pattern_1):
        m2 = re.search(pat, text)
        if m2:
            return m2.group('act')
    return None


def extract_coord(response: str) -> tuple[list[int], bool]:
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r"\[(?P<x>\d+),\s*(?P<y>\d+)\]"
    m = re.search(answer_tag_pattern, response, re.DOTALL)
    if not m:
        return [0, 0], False
    text = m.group(1)
    m2 = re.search(bbox_pattern, text)
    if not m2:
        return [0, 0], False
    return [int(m2.group('x')), int(m2.group('y'))], True


def extract_bbox(response: str) -> tuple[list[int], bool]:
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r"\[(?P<x1>\d+),\s*(?P<y1>\d+),\s*(?P<x2>\d+),\s*(?P<y2>\d+)\]"
    m = re.search(answer_tag_pattern, response, re.DOTALL)
    if not m:
        return [0, 0, 0, 0], False
    text = m.group(1)
    m2 = re.search(bbox_pattern, text)
    if not m2:
        return [0, 0, 0, 0], False
    return [int(m2.group('x1')), int(m2.group('y1')), int(m2.group('x2')), int(m2.group('y2'))], True

# --- Reward functions ---
def accuracy_reward_coord(completions, solution, scales, **kwargs):
    print("completions:", completions,
          "\n solution:", solution)
    # completions: List[str]
    rewards = []
    for content, gt_bbox, scale in zip(completions, solution, scales):
        reward = 0.0
        try:
          (col, row), ok = extract_coord(content)
          x_min, y_min, width, height = gt_bbox
          # check if predicted point lies within the bbox
          if (x_min <= col < x_min + width) and (y_min <= row < y_min + height):
              reward = 1.0
              print("succcess!")
        except:
            pass
        rewards.append(reward)
    return rewards