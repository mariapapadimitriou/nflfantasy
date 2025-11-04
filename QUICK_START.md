# Quick Start Guide - Django NFL Predictions App

## Running the Application

1. **Start the Django development server:**
   ```bash
   python manage.py runserver
   ```

2. **Open your browser:**
   Navigate to `http://localhost:8000`

3. **Use the application:**
   - Enter a season (e.g., 2025) and week (e.g., 9)
   - Click "Load Data" to load NFL data
   - Click "Train Model" to train a prediction model
   - Click "Predict Week" to generate predictions
   - Click "Export CSV" to download results

## Project Structure

```
touchdowns/
├── nfl_predictions/          # Django project
│   ├── settings.py           # Django settings
│   └── urls.py               # Main URL routing
├── predictions/              # Django app
│   ├── views.py              # API endpoints
│   ├── urls.py               # App URL routing
│   └── templates/            # HTML templates
├── static/                   # Static files
│   ├── css/style.css         # Styles
│   └── js/main.js            # JavaScript
├── predictions/              # Django app
│   ├── ml_model.py           # ML model
│   ├── data_manager.py       # Data loading
│   ├── config.py             # Configuration
│   └── models.py             # Database models
├── db.sqlite3                # SQLite database
```

## Key Features

- ✅ RESTful API endpoints for all operations
- ✅ Session-based data storage
- ✅ Modern, responsive UI
- ✅ Real-time status updates
- ✅ CSV export functionality
- ✅ Integration with existing model code

## API Endpoints

- `GET /` - Homepage
- `POST /api/load-data/` - Load NFL data
- `POST /api/train-model/` - Train model
- `POST /api/retrain-model/` - Retrain model
- `POST /api/predict-week/` - Generate predictions
- `POST /api/export-predictions/` - Export CSV
- `GET /api/check-model/` - Check if model exists

## Notes

- The app uses Django sessions to store data between requests
- Model files and historical data are stored in SQLite database (`db.sqlite3`)
- Training can take 5-10 minutes due to hyperparameter optimization
- All existing model and data management code is reused

For more details, see `README_DJANGO.md`.

