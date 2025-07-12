FROM nginx:alpine

# Copy static files
COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/
COPY status-handler.js /usr/share/nginx/html/
COPY shared/ /usr/share/nginx/html/shared/

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 8080

# Start nginx in foreground mode
CMD ["nginx", "-g", "daemon off;"] 