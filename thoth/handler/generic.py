import os.path


class Handler:
    def __init__(self, name):
        super().__init__()
        self.name = name

        cwd = os.path.abspath(os.path.dirname(__file__))
        self.text_path = os.path.join(cwd, f"../../text/{name}")

    def get_intro(self):
        with open(f"{self.text_path}/intro.md", "r") as f:
            return f.read()

    def render_playground(self):
        pass
