# Mental Health Chatbot - Weeks 5-6 Implementation
# Enhanced UI + Voice Features
# File: mental_health_chatbot_enhanced.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from textblob import TextBlob
import random
import plotly.express as px
import plotly.graph_objects as go
import speech_recognition as sr
import pyttsx3
import threading
import base64
import time

# Week 5: Enhanced UI Configuration
class UIConfig:
    """Enhanced UI configuration with colors, emojis, and themes"""
    
    # Color schemes for different emotions
    EMOTION_COLORS = {
        'very_positive': '#2E8B57',  # Sea Green
        'positive': '#90EE90',      # Light Green
        'neutral': '#FFD700',       # Gold
        'negative': '#FFA500',      # Orange
        'very_negative': '#FF6347'  # Tomato
    }
    
    # Emoji mappings
    EMOTION_EMOJIS = {
        'very_positive': '😄',
        'positive': '😊',
        'neutral': '😐',
        'negative': '😔',
        'very_negative': '😢'
    }
    
    # CSS for enhanced styling
    CUSTOM_CSS = """
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .mood-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .positive-mood {
        border-left: 5px solid #2E8B57;
    }
    
    .negative-mood {
        border-left: 5px solid #FF6347;
    }
    
    .neutral-mood {
        border-left: 5px solid #FFD700;
    }
    
    .voice-control {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        cursor: pointer;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """

# Week 1-4: Existing Classes (Updated)
class Config:
    """Configuration and constants for the chatbot"""
    
    CRISIS_KEYWORDS = [
        'suicide', 'kill myself', 'end it all', 'hurt myself', 
        'no point living', 'want to die', 'self harm', 'suicidal'
    ]
    
    CRISIS_RESPONSE = """
    🚨 **I'm really concerned about what you're going through.**
    
    You're not alone, and there are people who want to help you right now.
    
    **🆘 Immediate Help:**
    • National Suicide Prevention Lifeline: **988** (US)
    • Crisis Text Line: Text **HOME** to **741741**
    • International: **befrienders.org**
    
    Please reach out to a mental health professional, trusted friend, or family member immediately.
    **Your life has value and meaning.** 💙
    
    If you're in immediate danger, please call emergency services (911).
    """
    
    MOOD_DATA_FILE = "mood_data.json"
    CHAT_HISTORY_FILE = "chat_history.json"
    VOICE_SETTINGS_FILE = "voice_settings.json"

class MoodTracker:
    """Enhanced mood tracking with better data management"""
    
    def __init__(self):
        self.mood_data = self.load_mood_data()
        self.chat_history = self.load_chat_history()
    
    def load_mood_data(self):
        if os.path.exists(Config.MOOD_DATA_FILE):
            with open(Config.MOOD_DATA_FILE, 'r') as f:
                return json.load(f)
        return []
    
    def load_chat_history(self):
        if os.path.exists(Config.CHAT_HISTORY_FILE):
            with open(Config.CHAT_HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    
    def save_mood_data(self):
        with open(Config.MOOD_DATA_FILE, 'w') as f:
            json.dump(self.mood_data, f, indent=2)
    
    def save_chat_history(self):
        with open(Config.CHAT_HISTORY_FILE, 'w') as f:
            json.dump(self.chat_history, f, indent=2)
    
    def log_mood(self, user_input, emotion, polarity_score, input_method="text"):
        mood_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'emotion': emotion,
            'polarity_score': polarity_score,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'input_method': input_method  # 'text' or 'voice'
        }
        self.mood_data.append(mood_entry)
        self.save_mood_data()
    
    def log_chat(self, user_message, bot_response, emotion, input_method="text"):
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'bot_response': bot_response,
            'emotion': emotion,
            'input_method': input_method
        }
        self.chat_history.append(chat_entry)
        self.save_chat_history()

class EmotionAnalyzer:
    """Enhanced emotion analysis with confidence scoring"""
    
    @staticmethod
    def analyze_emotion(text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.3:
            emotion = "very_positive"
        elif polarity > 0.1:
            emotion = "positive"
        elif polarity > -0.1:
            emotion = "neutral"
        elif polarity > -0.3:
            emotion = "negative"
        else:
            emotion = "very_negative"
        
        return {
            'emotion': emotion,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': abs(polarity),
            'emoji': UIConfig.EMOTION_EMOJIS.get(emotion, '😐')
        }

class ResponseGenerator:
    """Enhanced response generation with more contextual responses"""
    
    def __init__(self):
        self.response_templates = {
            'very_positive': [
                "I'm absolutely thrilled to hear you're feeling so wonderful! 🌟 Your joy is contagious!",
                "What an amazing energy you have today! ✨ Tell me more about what's bringing you this happiness!",
                "Your positivity is lighting up the room! 😄 I love hearing when you're doing so well!",
            ],
            'positive': [
                "It's wonderful to hear some good vibes from you! 😊 What's been going well?",
                "I'm so glad you're feeling better today! 🌈 How can we keep this positive momentum going?",
                "That's really encouraging to hear! 💫 What small victories are you celebrating?",
            ],
            'neutral': [
                "Thanks for checking in with me. 🤗 Sometimes neutral is perfectly okay too.",
                "I hear you. What's one thing that might add a little spark to your day? ⭐",
                "I'm here to listen to whatever is on your mind. 💭 What would be helpful right now?",
            ],
            'negative': [
                "I can really hear that you're going through something difficult. 💙 That takes courage to share.",
                "It sounds like you're carrying some heavy feelings right now. 🫂 You don't have to face this alone.",
                "I'm sorry you're experiencing this struggle. 🌧️ What's been weighing most heavily on your heart?",
            ],
            'very_negative': [
                "I'm deeply concerned about how much pain you're in right now. 🤗 Thank you for trusting me with these feelings.",
                "This sounds incredibly overwhelming and difficult. 💜 Please know that even in the darkest moments, there can be hope.",
                "You're going through something really intense. 🌙 Have you been able to reach out to anyone you trust about this?",
            ]
        }
        
        self.coping_strategies = {
            'negative': [
                "💡 **Grounding Technique**: Try the 5-4-3-2-1 method: 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste.",
                "💡 **Breathing Exercise**: Inhale for 4 counts, hold for 4, exhale for 6. This activates your calm response.",
                "💡 **Gentle Movement**: A short walk, stretching, or even just stepping outside can help shift your energy.",
                "💡 **Self-Compassion**: Speak to yourself as you would a good friend. You deserve kindness, especially from yourself."
            ],
            'very_negative': [
                "💡 **Immediate Support**: Please consider reaching out to a crisis helpline or mental health professional.",
                "💡 **Safety First**: Focus on staying safe right now. What's one small step you can take to feel more secure?",
                "💡 **Connection**: Try to connect with someone you trust - a friend, family member, or counselor.",
                "💡 **Present Moment**: Ground yourself in the here and now. You made it through difficult times before."
            ]
        }
    
    def generate_response(self, user_input, emotion_data):
        emotion = emotion_data['emotion']
        
        # Check for crisis keywords
        if self.is_crisis_message(user_input):
            return Config.CRISIS_RESPONSE
        
        # Select appropriate response template
        responses = self.response_templates.get(emotion, self.response_templates['neutral'])
        base_response = random.choice(responses)
        
        # Add coping strategy for negative emotions
        if emotion in ['negative', 'very_negative']:
            coping_tip = random.choice(self.coping_strategies[emotion])
            base_response += "\n\n" + coping_tip
        
        return base_response
    
    def is_crisis_message(self, text):
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in Config.CRISIS_KEYWORDS)

# Week 6: Voice Integration Classes
class VoiceManager:
    """Handles voice input and output functionality"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.setup_voice_settings()
    
    def setup_voice_settings(self):
        """Configure text-to-speech settings"""
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Try to use a female voice if available
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
        
        self.tts_engine.setProperty('rate', 180)  # Speed
        self.tts_engine.setProperty('volume', 0.9)  # Volume
    
    def listen_for_speech(self, timeout=5, phrase_time_limit=10):
        """Listen for speech input from microphone"""
        try:
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Listen for audio
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            return text, True
            
        except sr.WaitTimeoutError:
            return "No speech detected. Please try again.", False
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand what you said. Please try again.", False
        except sr.RequestError as e:
            return f"Speech recognition error: {str(e)}", False
        except Exception as e:
            return f"Error: {str(e)}", False
    
    def speak_text(self, text):
        """Convert text to speech"""
        try:
            # Clean text for better speech
            clean_text = self.clean_text_for_speech(text)
            
            # Speak in a separate thread to avoid blocking
            def speak():
                self.tts_engine.say(clean_text)
                self.tts_engine.runAndWait()
            
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
            return True
            
        except Exception as e:
            st.error(f"Text-to-speech error: {str(e)}")
            return False
    
    def clean_text_for_speech(self, text):
        """Clean text for better speech synthesis"""
        # Remove markdown formatting
        text = text.replace('**', '')
        text = text.replace('*', '')
        text = text.replace('#', '')
        text = text.replace('`', '')
        
        # Remove emojis for cleaner speech
        import re
        text = re.sub(r'[^\w\s.,!?;:\-\(\)]', '', text)
        
        # Replace certain patterns
        text = text.replace('💡', 'Tip:')
        text = text.replace('🆘', 'Emergency:')
        text = text.replace('•', ' ')
        
        return text

# Week 4: Enhanced Visualization
class MoodVisualizer:
    """Enhanced mood visualization with better charts and insights"""
    
    @staticmethod
    def create_enhanced_mood_timeline(mood_data):
        """Create enhanced timeline with input method tracking"""
        if not mood_data:
            return None
        
        df = pd.DataFrame(mood_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Create daily mood averages
        daily_mood = df.groupby('date').agg({
            'polarity_score': 'mean',
            'input_method': lambda x: x.mode()[0] if not x.empty else 'text'
        }).reset_index()
        
        fig = px.line(daily_mood, x='date', y='polarity_score', 
                     title='Your Mood Journey Over Time',
                     labels={'polarity_score': 'Mood Score', 'date': 'Date'})
        
        # Add input method information
        fig.add_scatter(x=daily_mood['date'], y=daily_mood['polarity_score'],
                       mode='markers',
                       marker=dict(
                           color=daily_mood['input_method'].map({'voice': 'red', 'text': 'blue'}),
                           size=8
                       ),
                       name='Input Method',
                       hovertemplate='<b>%{x}</b><br>Mood: %{y:.2f}<br>Method: %{marker.color}<extra></extra>')
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="Neutral")
        fig.update_layout(height=400, showlegend=True)
        
        return fig
    
    @staticmethod
    def create_input_method_analysis(mood_data):
        """Analyze mood patterns by input method"""
        if not mood_data:
            return None
        
        df = pd.DataFrame(mood_data)
        
        # Group by input method
        method_analysis = df.groupby('input_method').agg({
            'polarity_score': ['mean', 'count'],
            'emotion': lambda x: x.mode()[0] if not x.empty else 'neutral'
        }).round(2)
        
        method_analysis.columns = ['avg_mood', 'count', 'common_emotion']
        method_analysis = method_analysis.reset_index()
        
        fig = px.bar(method_analysis, x='input_method', y='avg_mood',
                    title='Mood Patterns by Input Method',
                    color='avg_mood',
                    text='count')
        
        fig.update_traces(texttemplate='%{text} entries', textposition='outside')
        fig.update_layout(height=300)
        
        return fig

# Main Enhanced Application
class EnhancedMentalHealthChatbot:
    """Enhanced chatbot with UI improvements and voice features"""
    
    def __init__(self):
        self.mood_tracker = MoodTracker()
        self.emotion_analyzer = EmotionAnalyzer()
        self.response_generator = ResponseGenerator()
        self.visualizer = MoodVisualizer()
        self.voice_manager = VoiceManager()
        
        # Initialize session state
        if 'voice_mode' not in st.session_state:
            st.session_state.voice_mode = False
        if 'listening' not in st.session_state:
            st.session_state.listening = False
    
    def run(self):
        """Run the enhanced Streamlit application"""
        st.set_page_config(
            page_title="Mental Health Support",
            page_icon="🤗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        st.markdown(UIConfig.CUSTOM_CSS, unsafe_allow_html=True)
        
        # Enhanced header
        st.markdown("""
        <div class="main-header">
            <h1>🤗 CuraMind - Your Mental Health Support Chatbot </h1>
            <p>Your personal emotional wellness companion with voice support</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar for controls
        self.create_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.enhanced_chat_interface()
        
        with col2:
            self.enhanced_mood_dashboard()
    
    def create_sidebar(self):
        """Create enhanced sidebar with voice controls"""
        st.sidebar.header("🎛️ Controls")
        
        # Voice mode toggle
        st.session_state.voice_mode = st.sidebar.toggle(
            "🎤 Voice Mode", 
            value=st.session_state.voice_mode,
            help="Enable voice input and output"
        )
        
        if st.session_state.voice_mode:
            st.sidebar.success("🎤 Voice mode enabled!")
            st.sidebar.info("Click 'Listen' to start voice input")
        else:
            st.sidebar.info("💬 Text mode active")
        
        # Quick stats
        st.sidebar.header("📊 Quick Stats")
        if self.mood_tracker.mood_data:
            total_entries = len(self.mood_tracker.mood_data)
            latest_mood = self.mood_tracker.mood_data[-1]['emotion']
            voice_entries = sum(1 for entry in self.mood_tracker.mood_data if entry.get('input_method') == 'voice')
            
            st.sidebar.metric("Total Check-ins", total_entries)
            st.sidebar.metric("Latest Mood", latest_mood.replace('_', ' ').title())
            st.sidebar.metric("Voice Entries", voice_entries)
        
        # Data management
        st.sidebar.header("🗂️ Data Management")
        if st.sidebar.button("📥 Export Data"):
            self.export_data()
        
        if st.sidebar.button("🗑️ Clear All Data"):
            self.clear_all_data()
    
    def enhanced_chat_interface(self):
        """Enhanced chat interface with voice support"""
        st.header("💬 Chat Interface")
        
        # Initialize messages
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your enhanced mental health support companion. I can now listen to your voice and speak back to you! How are you feeling today? 🤗"}
            ]
        
        # Display chat history with enhanced styling
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Input section
        input_col1, input_col2 = st.columns([3, 1])
        
        with input_col1:
            # Text input (always available)
            text_input = st.chat_input("Share what's on your mind..." if not st.session_state.voice_mode else "Type here or use voice input...")
        
        with input_col2:
            # Voice input button (only in voice mode)
            if st.session_state.voice_mode:
                if st.button("🎤 Listen", key="voice_input"):
                    self.handle_voice_input()
        
        # Handle text input
        if text_input:
            self.process_user_input(text_input, "text")
    
    def handle_voice_input(self):
        """Handle voice input processing"""
        with st.spinner("🎤 Listening... Please speak now"):
            try:
                # Listen for speech
                text, success = self.voice_manager.listen_for_speech(timeout=10, phrase_time_limit=15)
                
                if success:
                    st.success(f"🎤 I heard: \"{text}\"")
                    self.process_user_input(text, "voice")
                else:
                    st.error(f"🎤 {text}")
                    
            except Exception as e:
                st.error(f"Voice input error: {str(e)}")
    
    def process_user_input(self, user_input, input_method):
        """Process user input (text or voice) and generate response"""
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Analyze emotion
        emotion_data = self.emotion_analyzer.analyze_emotion(user_input)
        
        # Generate response
        response = self.response_generator.generate_response(user_input, emotion_data)
        
        # Log the interaction
        self.mood_tracker.log_mood(user_input, emotion_data['emotion'], emotion_data['polarity'], input_method)
        self.mood_tracker.log_chat(user_input, response, emotion_data['emotion'], input_method)
        
        # Add response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Speak response if in voice mode
        if st.session_state.voice_mode:
            with st.spinner("🔊 Speaking..."):
                self.voice_manager.speak_text(response)
    
    def enhanced_mood_dashboard(self):
        """Enhanced mood dashboard with better visualizations"""
        st.header("📊 Enhanced Mood Dashboard")
        
        if not self.mood_tracker.mood_data:
            st.info("Start chatting to see your mood patterns here! 🚀")
            return
        
        # Weekly summary with enhanced styling
        st.subheader("📅 This Week's Overview")
        weekly_summary = self.create_enhanced_weekly_summary()
        
        # Create mood card with appropriate styling
        mood_class = self.get_mood_class(self.mood_tracker.mood_data[-1]['emotion'])
        st.markdown(f"""
        <div class="mood-card {mood_class}">
            {weekly_summary}
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced mood timeline
        st.subheader("📈 Mood Timeline")
        timeline_fig = self.visualizer.create_enhanced_mood_timeline(self.mood_tracker.mood_data)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Input method analysis
        if any(entry.get('input_method') == 'voice' for entry in self.mood_tracker.mood_data):
            st.subheader("🎤 Voice vs Text Analysis")
            method_fig = self.visualizer.create_input_method_analysis(self.mood_tracker.mood_data)
            if method_fig:
                st.plotly_chart(method_fig, use_container_width=True)
        
        # Emotion distribution
        st.subheader("🌈 Emotion Distribution")
        emotion_fig = self.create_enhanced_emotion_distribution()
        if emotion_fig:
            st.plotly_chart(emotion_fig, use_container_width=True)
    
    def create_enhanced_weekly_summary(self):
        """Create enhanced weekly summary with emoji support"""
        df = pd.DataFrame(self.mood_tracker.mood_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Last 7 days
        week_ago = datetime.now() - timedelta(days=7)
        recent_data = df[df['timestamp'] >= week_ago]
        
        if recent_data.empty:
            return "No data from the past week. Start logging your moods! 🌟"
        
        avg_mood = recent_data['polarity_score'].mean()
        most_common_emotion = recent_data['emotion'].mode()[0]
        voice_count = sum(1 for entry in recent_data.to_dict('records') if entry.get('input_method') == 'voice')
        
        emoji = UIConfig.EMOTION_EMOJIS.get(most_common_emotion, '😐')
        
        summary = f"""
        **Your Week in Review:** {emoji}
        
        • **Average mood score:** {avg_mood:.2f}
        • **Most common emotion:** {most_common_emotion.replace('_', ' ').title()}
        • **Total check-ins:** {len(recent_data)}
        • **Voice interactions:** {voice_count}
        """
        
        if avg_mood > 0.2:
            summary += "\n\n🌟 **You've had a generally positive week!** Keep up the great work!"
        elif avg_mood < -0.2:
            summary += "\n\n💙 **It's been a challenging week.** Remember to be kind to yourself and reach out for support when needed."
        else:
            summary += "\n\n🌤️ **You've had a balanced week** with a mix of emotions, which is completely normal."
        
        return summary
    
    def create_enhanced_emotion_distribution(self):
        """Create enhanced emotion distribution chart"""
        df = pd.DataFrame(self.mood_tracker.mood_data)
        emotion_counts = df['emotion'].value_counts()
        
        fig = px.pie(
            values=emotion_counts.values, 
            names=emotion_counts.index,
            title='Distribution of Your Emotions',
            color=emotion_counts.index,
            color_discrete_map=UIConfig.EMOTION_COLORS
        )
        
        # Add emojis to labels
        fig.update_traces(
            textinfo='label+percent',
            texttemplate='%{label} %{percent}<br>%{value} times'
        )
        
        return fig
    
    def get_mood_class(self, emotion):
        """Get CSS class for mood styling"""
        if emotion in ['very_positive', 'positive']:
            return 'positive-mood'
        elif emotion in ['very_negative', 'negative']:
            return 'negative-mood'
        else:
            return 'neutral-mood'
    
    def export_data(self):
        """Export user data to CSV"""
        try:
            df = pd.DataFrame(self.mood_tracker.mood_data)
            csv = df.to_csv(index=False)
            
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="mood_data.csv">📥 Download CSV</a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)
            st.sidebar.success("Data exported successfully!")
            
        except Exception as e:
            st.sidebar.error(f"Export failed: {str(e)}")
    
    def clear_all_data(self):
        """Clear all stored data"""
        try:
            # Clear files
            for file in [Config.MOOD_DATA_FILE, Config.CHAT_HISTORY_FILE]:
                if os.path.exists(file):
                    os.remove(file)
            
            # Clear session state
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your enhanced mental health support companion. How are you feeling today? 🤗"}
            ]
            
            st.sidebar.success("All data cleared!")
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Clear failed: {str(e)}")

# Run the enhanced application
if __name__ == "__main__":
    try:
        chatbot = EnhancedMentalHealthChatbot()
        chatbot.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check your dependencies and try again.")