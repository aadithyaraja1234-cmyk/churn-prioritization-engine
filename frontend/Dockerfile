# Multi-stage: node builds the static bundle, a much smaller nginx image
# actually serves it - the node_modules/build toolchain never ships in the
# final image.
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Baked in at build time (Vite inlines import.meta.env.* into the bundle -
# there's no such thing as changing this at container-start time the way a
# backend env var works). Defaults to the same localhost:8000 client.js
# already falls back to, so a plain `docker build` with no --build-arg
# behaves exactly like local dev. Override via docker-compose.yml's
# build.args for a real deployment where the backend lives elsewhere.
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost/ >/dev/null || exit 1
