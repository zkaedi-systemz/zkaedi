"""
Renderers module - Beautiful visualizations

Where mathematics becomes art, where data tells stories.
"""

from .live_html import LiveHTMLRenderer, create_live_visualization

__all__ = [
    'LiveHTMLRenderer',
    'create_live_visualization',
]