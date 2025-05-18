"""
external_search.py: Gère la recherche externe via Wikipedia et autres sources
"""

import wikipedia
import requests
from typing import List, Dict, Any
import logging
from bs4 import BeautifulSoup
import json
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExternalSearchManager:
    """Gère les recherches externes via différentes sources."""
    
    def __init__(self):
        """Initialise le gestionnaire de recherche externe."""
        # Configurer Wikipedia
        wikipedia.set_lang("en")  # Langue par défaut
        
    def search_wikipedia(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Recherche dans Wikipedia.
        
        Args:
            query: La requête de recherche
            num_results: Nombre de résultats à retourner
            
        Returns:
            Liste de résultats avec texte et métadonnées
        """
        try:
            # Rechercher les pages
            search_results = wikipedia.search(query, results=num_results)
            results = []
            
            for title in search_results:
                try:
                    # Obtenir le contenu de la page
                    page = wikipedia.page(title)
                    results.append({
                        "text": page.content[:1000],  # Limiter la longueur
                        "metadata": {
                            "source": "Wikipedia",
                            "title": title,
                            "url": page.url
                        }
                    })
                except wikipedia.exceptions.DisambiguationError as e:
                    # Gérer les pages d'homonymie
                    continue
                except Exception as e:
                    logger.error(f"Error fetching Wikipedia page {title}: {str(e)}")
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error in Wikipedia search: {str(e)}")
            return []
    
    def search_web(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Recherche sur le web via DuckDuckGo.
        
        Args:
            query: La requête de recherche
            num_results: Nombre de résultats à retourner
            
        Returns:
            Liste de résultats avec texte et métadonnées
        """
        try:
            results = []
            # Encoder la requête pour l'URL
            encoded_query = urllib.parse.quote(query)
            search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            # Faire la requête
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(search_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraire les résultats
            search_results = soup.find_all('div', class_='result')
            
            for result in search_results[:num_results]:
                try:
                    title_elem = result.find('a', class_='result__title')
                    snippet_elem = result.find('a', class_='result__snippet')
                    
                    if title_elem and snippet_elem:
                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True)
                        
                        results.append({
                            "text": f"{title}\n{snippet}",
                            "metadata": {
                                "source": "Web",
                                "url": url,
                                "title": title
                            }
                        })
                except Exception as e:
                    logger.error(f"Error processing search result: {str(e)}")
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
    
    def search_all_sources(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Recherche dans toutes les sources disponibles.
        
        Args:
            query: La requête de recherche
            num_results: Nombre de résultats par source
            
        Returns:
            Liste combinée de résultats
        """
        # Rechercher dans Wikipedia
        wiki_results = self.search_wikipedia(query, num_results)
        
        # Rechercher sur le web
        web_results = self.search_web(query, num_results)
        
        # Combiner les résultats
        return wiki_results + web_results 