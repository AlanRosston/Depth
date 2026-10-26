# Static tmux 3.4 (image hd/tmux-static, depth.taskenv.TMUX_STATIC_IMAGE): copied into task images whose
# distribution has no tmux or only tmux 1.x (depth/taskenv.py).
#   docker build -t hd/tmux-static -f depth/tools/tmux-static.Dockerfile depth/tools      # a few minutes
FROM alpine:3.20
RUN apk add --no-cache build-base automake autoconf pkgconf bison libevent-dev libevent-static \
    ncurses-dev ncurses-static curl
RUN curl -LsSf https://github.com/tmux/tmux/releases/download/3.4/tmux-3.4.tar.gz -o /tmp/t.tgz \
    && tar xzf /tmp/t.tgz -C /tmp
WORKDIR /tmp/tmux-3.4
RUN ./configure --enable-static CFLAGS="-static" LDFLAGS="-static" && make -j4 && cp tmux /tmux && strip /tmux && /tmux -V
