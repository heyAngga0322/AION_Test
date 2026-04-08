# Stage 1: Build Angular App
FROM node:22-alpine AS build

WORKDIR /app
COPY chat-ui/package*.json ./
RUN npm install

COPY chat-ui/ .
RUN npm run build -- --configuration production

# Stage 2: Serve via Nginx
FROM nginx:alpine

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy build artifacts
# Note: Angular 18+ outputs to dist/project-name/browser
COPY --from=build /app/dist/chat-ui/browser /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
