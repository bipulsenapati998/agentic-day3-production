import re
import yaml
import logging
from typing import Final
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

INJECTION_PATTERNS: Final[list[str]] = [
	r"ignore (your |all |previous )?instructions",
	r"system prompt.*disabled",
	r"new role",
	r"repeat.*system prompt",
	r"jailbreak",
]


def detect_injection(user_input: str) -> bool:
	"""Return True if the input looks like a prompt injection attempt."""
	text = user_input.lower()
	for pattern in INJECTION_PATTERNS:
		if re.search(pattern, text):
			return True
	return False