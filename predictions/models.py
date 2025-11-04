"""
Django models for NFL Touchdown Predictions
"""
from django.db import models
import json


class TrainingData(models.Model):
    """Store historical training data"""
    season = models.IntegerField()
    week = models.IntegerField()
    player_id = models.CharField(max_length=50)
    player_name = models.CharField(max_length=200)
    team = models.CharField(max_length=10)
    position = models.CharField(max_length=10)
    against = models.CharField(max_length=10)
    touchdown = models.IntegerField()
    played = models.IntegerField()
    report_status = models.CharField(max_length=20)
    
    # Feature columns stored as JSON for flexibility
    features = models.JSONField(default=dict)
    
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'training_data'
        indexes = [
            models.Index(fields=['season', 'week']),
            models.Index(fields=['player_id']),
            models.Index(fields=['season', 'week', 'player_id']),
        ]
        unique_together = [['season', 'week', 'player_id']]


class MLModel(models.Model):
    """Store trained ML models"""
    season = models.IntegerField()
    week = models.IntegerField()
    
    # Model file stored as binary
    model_file = models.BinaryField()
    
    # Preprocessor files stored as binary
    imputer_file = models.BinaryField()
    scaler_file = models.BinaryField()
    encoder_file = models.BinaryField()
    
    # Model metadata
    created_at = models.DateTimeField(auto_now_add=True)
    training_records = models.IntegerField(null=True)
    
    class Meta:
        db_table = 'ml_models'
        indexes = [
            models.Index(fields=['season', 'week']),
        ]
        unique_together = [['season', 'week']]


class Prediction(models.Model):
    """Store predictions for players"""
    season = models.IntegerField()
    week = models.IntegerField()
    player_id = models.CharField(max_length=50)
    player_name = models.CharField(max_length=200)
    team = models.CharField(max_length=10)
    position = models.CharField(max_length=10)
    against = models.CharField(max_length=10)
    probability = models.FloatField()
    played = models.CharField(max_length=1, null=True)
    actual_touchdown = models.IntegerField(null=True)
    report_status = models.CharField(max_length=20, null=True)
    
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'predictions'
        indexes = [
            models.Index(fields=['season', 'week']),
            models.Index(fields=['player_id']),
            models.Index(fields=['season', 'week', 'player_id']),
        ]
        unique_together = [['season', 'week', 'player_id']]


class DataCache(models.Model):
    """Store cached computed statistics"""
    cache_key = models.CharField(max_length=200, unique=True)
    cache_data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'data_cache'
        indexes = [
            models.Index(fields=['cache_key']),
        ]
