LLAVA_PROMPT = """
Tu es un expert en vision par ordinateur.

Analyse UNIQUEMENT l'objet visible dans l'image.
Ne devine PAS la marque si elle n'est pas clairement visible.
Si un attribut n'est pas identifiable, mets null.

Retourne STRICTEMENT un JSON valide avec :
- type
- color
- brand
- material
- description

Exemple :
{
  "type": "chaussure",
  "color": "blanc",
  "brand": null,
  "material": "cuir",
  "description": "chaussure de sport basse"
}
"""

LLAVA_PROMPT_WITH_ITEM = """
Tu es un expert en vision par ordinateur.

L'objet à analyser est un(e) : {item_name}

Analyse UNIQUEMENT cet objet visible dans l'image.
Utilise le nom fourni pour affiner ta description et te concentrer sur les détails pertinents.
Ne devine PAS la marque si elle n'est pas clairement visible.
Si un attribut n'est pas identifiable, mets null.

Retourne STRICTEMENT un JSON valide avec :
- type
- color
- brand
- material
- description (détaillée et spécifique à ce type d'objet)

Exemple pour un téléphone :
{{
  "type": "téléphone de bureau",
  "color": "noir",
  "brand": "Cisco",
  "material": "plastique",
  "description": "téléphone IP avec écran LCD, 12 touches programmables et port casque"
}}
"""
