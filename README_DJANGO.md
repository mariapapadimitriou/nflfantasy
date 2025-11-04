# NFL Touchdown Predictions - Django Web Application

A Django web application for NFL touchdown predictions using machine learning (XGBoost).

## Features

- **Data Loading**: Load NFL data for specific seasons and weeks with intelligent caching
- **Model Training**: Train XGBoost models with Optuna hyperparameter optimization
- **Predictions**: Generate touchdown predictions for players in a given week
- **Export**: Download predictions as CSV files
- **Modern UI**: Clean, responsive interface with dark theme

## Setup

### Prerequisites

- Python 3.8+
- All dependencies from `requirements.txt`

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run database migrations (if needed):
```bash
python manage.py migrate
```

3. Collect static files:
```bash
python manage.py collectstatic
```

4. Run the development server:
```bash
python manage.py runserver
```

5. Open your browser and navigate to:
```
http://localhost:8000
```

## Usage

### Workflow

1. **Load Data**: Enter a season and week, then click "Load Data"
   - This loads historical data for training and current week data for predictions
   - Enable "Force Reload Data" to bypass cache and fetch fresh data

2. **Train Model**: Click "Train Model" to train a new model for the specified season/week
   - Training uses Optuna for hyperparameter optimization (may take 5-10 minutes)
   - If a model already exists, use "Retrain Model" to overwrite it

3. **Predict Week**: Click "Predict Week" to generate predictions
   - Requires a trained model for the season/week
   - Displays predictions in a sortable table

4. **Export CSV**: Download predictions as a CSV file

### API Endpoints

The application exposes the following API endpoints:

- `POST /api/load-data/` - Load NFL data
- `POST /api/train-model/` - Train a new model
- `POST /api/retrain-model/` - Retrain/overwrite existing model
- `POST /api/predict-week/` - Generate predictions
- `POST /api/export-predictions/` - Export predictions as CSV
- `GET /api/check-model/` - Check if model exists

### Project Structure

```
touchdowns/
├── nfl_predictions/          # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── predictions/              # Django app
│   ├── views.py              # API views
│   ├── urls.py               # URL routing
│   └── templates/            # HTML templates
├── static/                   # Static files (CSS, JS)
│   ├── css/
│   └── js/
├── predictions/              # Django app (all ML code here)
│   ├── ml_model.py           # ML model training/prediction
│   ├── data_manager.py       # Data loading and caching
│   ├── data_source.py        # NFL data source interface
│   ├── config.py             # Configuration
│   ├── utils.py              # Utility functions
│   └── models.py             # Django database models
├── db.sqlite3                # SQLite database (models & data)
└── manage.py                 # Django management script
```

## Differences from Dash App

The Django version provides:

- **Better structure**: Follows Django conventions
- **RESTful API**: JSON-based API endpoints
- **Session management**: Data stored in Django sessions
- **Production-ready**: Can be deployed to production servers
- **Extensible**: Easy to add authentication, database models, etc.

## Notes

- The application uses Django sessions to store loaded data between requests
- Model files and historical data are stored in SQLite database (`db.sqlite3`)
- Data caching is handled by `data_manager.py` with database storage
- Training models can take several minutes due to hyperparameter optimization

## Troubleshooting

### CSRF Token Errors

If you encounter CSRF token errors, ensure:
- Cookies are enabled in your browser
- The CSRF middleware is enabled in `settings.py`

### Model Not Found Errors

- Ensure you've trained a model for the specified season/week before making predictions
- Models are stored in the SQLite database, check using Django admin or database queries

### Data Loading Issues

- Check your internet connection (data is fetched from nflreadpy)
- Verify the season/week combination is valid
- Check the console for detailed error messages

## Development

To extend the application:

1. **Add new views**: Edit `predictions/views.py`
2. **Add new URLs**: Edit `predictions/urls.py`
3. **Modify templates**: Edit files in `predictions/templates/predictions/`
4. **Update styles**: Edit `static/css/style.css`
5. **Add JavaScript**: Edit `static/js/main.js`

## License

Same as the main project.

