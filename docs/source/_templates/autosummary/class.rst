{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   
   {% block init %}
   {%- if '__init__' in all_methods %}
   .. automethod:: __init__
   {%- endif -%}
   {% endblock %}
   
   {% block methods %}   
   
   .. HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
      .. autosummary::
         :toctree:
      {% for item in all_methods %}
         {%- if not item.startswith('_') or item in ['__call__', '_call', '_apply', '_lincomb', '_multiply', '_divide', '_dist', '_norm', '_inner', '__contains__', '__eq__', '__getitem__', '__setitem__'] %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
   {% endblock %}

   {% block attributes %} 
   .. HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
      .. autosummary::
         :toctree:
      {% for item in all_attributes %}
         {%- if not item.startswith('_') %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
   {% endblock %}
   