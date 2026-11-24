Référence de l'API
====================

Documentation générée automatiquement à partir du code source [#f1]_.

.. toctree::
   :titlesonly:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}

.. [#f1] Générée par `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_
