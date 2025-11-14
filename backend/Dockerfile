# Stage 1: The final runtime stage
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# --- DEPENDENCY INSTALLATION (as root) ---
# Copy the requirements file
COPY requirements.txt .

# Install dependencies using pip as root. This avoids all permission errors.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- APPLICATION SETUP (as non-root) ---
# Now, create the non-root user
RUN addgroup --system app && adduser --system --group app

# Copy the application code and set correct ownership
COPY --chown=app:app backend/app /app

# Switch to the non-root user for running the application
USER app

# Expose the port and define the runtime command
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]