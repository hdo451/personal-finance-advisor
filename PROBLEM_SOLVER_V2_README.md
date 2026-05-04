# 🧮 Solucionador de Problemas Financieros v2

Módulo **completamente aislado e independiente** para analizar y resolver problemas financieros complejos mediante una arquitectura de 4 etapas.

## 🎯 Arquitectura: 4 Etapas Secuenciales

### Etapa 1: Ingreso del Problema
- El usuario escribe el enunciado en **lenguaje natural**
- El problema puede contener **una o múltiples opciones/alternativas** a comparar
- **Ejemplo:** _"Necesito un crédito de $100,000 a 5% anual vs arrendar por $2,000 mensuales"_

### Etapa 2: Extracción y Validación Humana
- El **LLM parsea** el enunciado e identifica todas las opciones presentes
- Para cada opción, **extrae los parámetros** detectados
- Los valores **no mencionados** se asumen con estándares razonables
- Valores supuestos se **destacan visualmente** (🔶 badge)
- El usuario puede **editar cualquier valor** directamente
- **Solo con aprobación explícita** se avanza al cálculo

### Etapa 3: Selección de Columnas de Salida
- El usuario selecciona mediante **checkboxes** qué variables desea ver
- Disponibles: valor presente, futuro, cuota, tasa, períodos, saldo, amortización, intereses, depreciación, etc.
- **Solo se calculan** las columnas seleccionadas

### Etapa 4: Tabla de Resultados
- **Tabla comparativa** donde:
  - Cada **columna** = una opción/alternativa
  - Cada **fila** = una variable financiera seleccionada
- Celdas con valores supuestos se **destacan visualmente**
- **Descarga como CSV** disponible

## 🏗️ Arquitectura Técnica

```
streamlit_problem_solver_v2.py    ← UI (4 etapas)
    ↓
utils/
    ├── llm_problem_parser.py      ← Parser LLM (interpreta, NO calcula)
    └── financial_calculator_v2.py ← Motor determinístico (calcula)
```

## 🔍 Rol del LLM (Parser)

**Responsabilidades:**
1. Detectar cuántas opciones contiene el problema
2. Asignar correctamente cada dato extraído a su opción
3. Identificar variables faltantes
4. Proponer valores supuestos razonables
5. Devolver JSON estructurado con flags `assumed: true/false`

**NO calcula.** Solo interpreta, clasifica y estructura.

## ⚙️ Motor de Cálculo

**Usa librerías determinísticas:**
- `numpy-financial` para funciones financieras
- `scipy.optimize` para soluciones numéricas
- Resultados **exactos y reproducibles**

## 📊 Cobertura de Problemas Financieros

✅ Valor presente y valor futuro  
✅ Tasas de interés (nominal, efectiva, conversión)  
✅ Valor de cuotas / amortización (francés, alemán, americano)  
✅ Número de períodos  
✅ Depreciación (línea recta, doble saldo, suma de dígitos)  
✅ Gastos y flujos de caja simples  
✅ TIR (IRR) y VAN (NPV)  

## 🚀 Cómo Usar

### Versión UI (Recomendada)
```bash
source .venv/bin/activate
streamlit run streamlit_problem_solver_v2.py
```

Se abrirá en `http://localhost:8501` con interfaz interactiva de 4 etapas.

### Uso Programático

```python
from utils.llm_problem_parser import LLMProblemParser
from utils.financial_calculator_v2 import FinancialCalculator

# Paso 1: Parsear el problema
parser = LLMProblemParser()
problem = "Crédito de $50,000 a 8% anual por 5 años"
parsed = parser.parse_problem(problem)

# Paso 2: Acceder a opciones y parámetros
for option in parsed["options"]:
    print(f"Opción: {option['name']}")
    print(f"Parámetros: {option['parameters']}")

# Paso 3: Calcular usando el motor
result = FinancialCalculator.calculate_compound_interest(
    principal=50000,
    rate=0.08,
    periods=5
)
print(f"Valor Futuro: ${result['fv']:.2f}")
```

## 📦 JSON Structure del Parser

```json
{
  "success": true,
  "problem_type": "amortization",
  "extraction_confidence": 0.92,
  "options": [
    {
      "name": "Opción A",
      "parameters": {
        "principal": {
          "value": 50000,
          "unit": "USD",
          "assumed": false
        },
        "rate": {
          "value": 0.08,
          "unit": "annual",
          "assumed": false
        },
        "periods": {
          "value": 60,
          "unit": "months",
          "assumed": true
        }
      }
    }
  ],
  "assumptions_made": [
    "periods = 60 meses (5 años estándar)"
  ],
  "missing_critical_parameters": []
}
```

## 🧪 Pruebas

```bash
source .venv/bin/activate
python -m pytest tests/test_problem_solver_v2.py -v
```

**14 pruebas unitarias** validando:
- Parser LLM
- Cálculos determinísticos
- Integración completa

## 🔐 Configuración

La API key se lee **exclusivamente desde `.env`**:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
chmod 600 .env
```

**Nunca se hardcodea** ni se solicita al usuario durante ejecución.

## 📝 Valores Supuestos (Defaults)

Cuando el usuario omite datos necesarios:

| Parámetro | Valor por Defecto | Unidad |
|-----------|-------------------|--------|
| Períodos | 12 | meses |
| Tipo de Tasa | monthly | - |
| Depreciación | straight_line | - |
| Amortización | french | - |

Todos aparecen **claramente marcados** como "🔶 SUPUESTO" en la UI para que el usuario los valide.

## 🎨 Criterios de Diseño

✅ **Simple en estructura** — 4 etapas secuenciales claras  
✅ **Robusto en cobertura** — múltiples tipos de problemas  
✅ **Visual diferenciado** — supuestos vs explícitos  
✅ **Modular e independiente** — sin dependencias del resto del código  
✅ **Determinístico** — resultados reproducibles  
✅ **Tolerante a errores** — manejo claro cuando fallan los datos  

## 🔮 Próximos Pasos

1. ✅ Validar con casos reales
2. ⏳ Integrar tabla de amortización completa en resultados
3. ⏳ Agregar gráficos comparativos (Plotly)
4. ⏳ Exportar a PDF con reportes
5. ⏳ Integrar con módulo actual de "problemas cotidianos"

---

**Estado:** ✅ Prototipo funcional con tests pasados  
**Última actualización:** 2026-05-04  
**Modulo independiente:** Completamente aislado del resto de `streamlit_app.py`
