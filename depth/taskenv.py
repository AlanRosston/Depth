"""Task directories and images: installing tmux at image build time, normalizing task.toml, setting docker_image."""
import re
import subprocess
import tempfile

# tmux installation command of terminus-2's agent setup, run at image build time
TMUX = ("if command -v tmux >/dev/null 2>&1; then :; "
        "elif command -v apt-get >/dev/null 2>&1; then DEBIAN_FRONTEND=noninteractive apt-get update "
        "&& DEBIAN_FRONTEND=noninteractive apt-get install -y tmux; "
        "elif command -v apk >/dev/null 2>&1; then apk add --no-cache tmux; "
        "elif command -v dnf >/dev/null 2>&1; then dnf install -y tmux; "
        "elif command -v yum >/dev/null 2>&1; then yum install -y tmux; fi; true")
# static tmux binary from the image TMUX_STATIC_IMAGE (depth/tools/tmux-static.Dockerfile), copied if tmux is missing
TMUX_STATIC_IMAGE = "hd/tmux-static"
TMUX_FALLBACK = ("RUN --mount=type=bind,from=%s,source=/tmux,target=/tmp/hd-tmux "
                 "if ! command -v tmux >/dev/null 2>&1; then cp /tmp/hd-tmux /usr/local/bin/tmux "
                 "&& chmod 0755 /usr/local/bin/tmux; fi" % TMUX_STATIC_IMAGE)


def prebuilt_image(task_dir):
    m = re.search(r'^docker_image\s*=\s*"([^"]+)"', (task_dir / "task.toml").read_text(), re.M)
    return m.group(1) if m else None


def tmux_image(base):
    """Image exp-tmux/<name>:tmux: `base` with a tmux layer, built if missing; `base` if the build fails."""
    tag = "exp-tmux/%s:tmux" % re.sub(r"[^a-z0-9_.-]", "_", base.lower())
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0:
        return tag
    if subprocess.run(["docker", "image", "inspect", base], capture_output=True).returncode:
        subprocess.run(["docker", "pull", "-q", base], capture_output=True, timeout=1800)
    user = subprocess.run(["docker", "inspect", "--format", "{{.Config.User}}", base],
                          capture_output=True, text=True).stdout.strip()
    df = "FROM %s\nUSER root\nRUN %s\n%s\n%s" % (base, TMUX, TMUX_FALLBACK, ("USER %s\n" % user) if user else "")
    with tempfile.TemporaryDirectory() as ctx:
        r = subprocess.run(["docker", "build", "-q", "-t", tag, "-f", "-", ctx], input=df,
                           capture_output=True, text=True, timeout=1800)
    if r.returncode:
        print(f"could not add a tmux layer, using {base}: {r.stderr.strip()[-200:]}", flush=True)
        return base
    return tag


HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def dockerfile_users(text):
    """USER instructions of a Dockerfile, in order, skipping heredoc bodies."""
    users, end = [], None
    for ln in text.splitlines():
        if end is not None:
            if ln.strip() == end:
                end = None
            continue
        s = ln.strip()
        if s.upper().startswith("USER ") and len(s.split()) > 1:
            users.append(s.split()[1])
        m = HEREDOC.search(ln)
        if m:
            end = m.group(2)
    return users


def bake_tmux(dst):
    """Install tmux at build time in the task copy `dst`: a tmux layer for a prebuilt image, extra lines for a Dockerfile."""
    base = prebuilt_image(dst)
    if base:
        toml = dst / "task.toml"
        toml.write_text(re.sub(r'^docker_image\s*=\s*"[^"]+"', 'docker_image = "%s"' % tmux_image(base),
                               toml.read_text(), count=1, flags=re.M))
        return
    df = dst / "environment" / "Dockerfile"
    if not df.exists():
        return
    text = df.read_text().rstrip("\n")
    users = dockerfile_users(text)
    tail = "\n\n# tmux for terminus-2, installed at build time; a static tmux is the fallback\n"
    if users and users[-1] not in ("root", "0"):
        tail += "USER root\nRUN %s\n%s\nUSER %s\n" % (TMUX, TMUX_FALLBACK, users[-1])
    else:
        tail += "RUN %s\n%s\n" % (TMUX, TMUX_FALLBACK)
    df.write_text(text + tail)


def normalize_toml(text, test_sh=""):
    """task.toml with `memory` / `memory_mb` (and `storage` / `storage_mb`) set to the larger of the two when they
    disagree, and `allow_internet = true` when test_sh downloads packages."""
    if re.search(r"curl|uvx|pip install|apt-get", test_sh or ""):
        text = re.sub(r"^allow_internet\s*=\s*false", "allow_internet = true", text, flags=re.M)

    def unit_mb(s):
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([GMK])i?B?\s*", s or "", re.I)
        return int(float(m.group(1)) * {"G": 1024, "M": 1, "K": 1 / 1024}[m.group(2).upper()]) if m else None

    for key in ("memory", "storage"):
        m1 = re.search(rf'^{key}\s*=\s*"([^"]+)"', text, re.M)
        m2 = re.search(rf"^{key}_mb\s*=\s*(\d+)", text, re.M)
        vals = [v for v in (unit_mb(m1.group(1)) if m1 else None, int(m2.group(1)) if m2 else None) if v]
        if len(vals) < 2 or vals[0] == vals[1]:
            continue
        mb = max(vals)
        text = re.sub(rf'^{key}\s*=\s*"[^"]+"', f'{key} = "{mb // 1024}G"' if mb % 1024 == 0 else f'{key} = "{mb}M"',
                      text, flags=re.M)
        text = re.sub(rf"^{key}_mb\s*=\s*\d+", f"{key}_mb = {mb}", text, flags=re.M)
    return text


def set_docker_image(toml_text, tag):
    """Set docker_image in the [environment] section of a task.toml (replacing an existing one)."""
    t = re.sub(r'^docker_image\s*=.*\n', "", toml_text, flags=re.M)
    if re.search(r"^\[environment\]", t, re.M):
        return re.sub(r"^\[environment\]\n", f'[environment]\ndocker_image = "{tag}"\n', t, count=1, flags=re.M)
    return t.rstrip("\n") + f'\n\n[environment]\ndocker_image = "{tag}"\n'
