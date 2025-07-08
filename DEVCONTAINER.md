# TreePlanter - Devcontainer Setup

This project includes a complete devcontainer configuration for consistent development across different environments.

## 🐳 Devcontainer Features

### Included Tools & Extensions
- **Python 3.11** with pip, pylint
- **VS Code Extensions**: Python, Jupyter, JSON, auto-formatting
- **Flask Development Server** with auto-reload
- **Git & GitHub CLI** for version control
- **Debugging Support** with breakpoints and step-through debugging

### Pre-configured Environment
- **Flask App**: `app.py`
- **Development Mode**: Debug enabled, auto-reload on file changes
- **Port Forwarding**: Port 5000 automatically forwarded to host
- **Dependencies**: All Python packages auto-installed from `requirements.txt`

## 🚀 Quick Start

### Option 1: Using VS Code
1. **Install Prerequisites**:
   - [Visual Studio Code](https://code.visualstudio.com/)
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - [Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Open in Devcontainer**:
   ```bash
   # Clone and open the project
   git clone <repository-url>
   cd TreePlanter
   code .
   ```
   - VS Code will detect the devcontainer configuration
   - Click "Reopen in Container" when prompted
   - Or use Command Palette: `Dev Containers: Reopen in Container`

3. **Start the Application**:
   - **Quick Start**: Run `./start.sh` in the terminal
   - **Debug Mode**: Use VS Code's Run & Debug panel → "Python: Flask App"
   - **Manual**: `flask run --host=0.0.0.0 --port=5000`

### Option 2: Using Docker CLI
```bash
# Build and run the devcontainer
docker build -t treeplanter-dev .devcontainer/
docker run -p 5000:5000 -v $(pwd):/workspace treeplanter-dev
```

## 🌐 Accessing the Application

Once running, the TreePlanter app will be available at:
- **Local**: http://localhost:5000
- **VS Code**: Click the "Open in Browser" notification
- **Port Panel**: VS Code's Ports panel will show forwarded port 5000

## 🔧 Development Workflow

### Debugging
- Set breakpoints in VS Code
- Use "Python: Flask App" launch configuration
- Step through code with full variable inspection

### Code Formatting
- **Auto-format on save** with Ruff formatter
- **Linting** with Pylint
- **Import organization** on save

### Testing
- Run tests with `pytest`
- VS Code Test Explorer integration
- Coverage reports available

### Hot Reload
- File changes automatically reload the Flask server
- No need to restart for Python/HTML/CSS changes
- Static files served automatically

## 📁 Project Structure
```
TreePlanter/
├── .devcontainer/
│   └── devcontainer.json      # Devcontainer configuration
├── .vscode/
│   ├── launch.json           # Debug configurations
│   └── settings.json         # VS Code workspace settings
├── templates/
│   └── index.html           # Flask HTML template
├── static/                  # Static assets (auto-created)
├── app.py                   # Main Flask application
├── requirements.txt         # Python dependencies
├── .env.development        # Development environment variables
└── start.sh                # Quick start script
```

## 🛠 Customization

### Adding Python Packages
1. Add packages to `requirements.txt`
2. Rebuild container or run `pip install -r requirements.txt`

### VS Code Extensions
Edit `.devcontainer/devcontainer.json` to add more extensions:
```json
"extensions": [
    "ms-python.python",
    "your-new-extension-id"
]
```

### Environment Variables
Edit `.env.development` for additional environment configuration.

## 📝 Troubleshooting

### Container Won't Start
- Ensure Docker Desktop is running
- Check Docker resources (memory/disk space)
- Rebuild container: Command Palette → "Dev Containers: Rebuild Container"

### Port Already in Use
- Stop other services using port 5000
- Or modify port in `devcontainer.json` and `app.py`

### Dependencies Issues
- Rebuild container to reinstall all dependencies
- Check `requirements.txt` for version conflicts

### Flask App Not Loading
- Check the terminal for error messages
- Ensure `app.py` is in the workspace root
- Verify environment variables in `.env.development`

## 🔍 Additional Resources

- [VS Code Dev Containers Documentation](https://code.visualstudio.com/docs/remote/containers)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)

## 🎯 Next Steps

1. **Start coding** - The environment is ready!
2. **Set breakpoints** - Use VS Code's debugging features
3. **Add tests** - Create test files for your algorithms
4. **Deploy** - Use the containerized app for production deployment

Happy coding! 🌲✨
