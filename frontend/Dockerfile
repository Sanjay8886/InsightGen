# Use a Node.js base image
FROM node:20-alpine

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
# We copy this first to leverage Docker's build cache
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Fix ownership and switch to the non-root 'node' user for security 
# and better compatibility with Docker volume mounting on host systems.
RUN chown -R node:node /app
USER node

# Expose the port where Vite runs
EXPOSE 3000

# Start the Vite development server. 
# --host 0.0.0.0 is critical for Docker to expose the port correctly.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
