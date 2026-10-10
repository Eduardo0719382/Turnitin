# Detector de IA (calibrado con reportes de Turnitin)

Analiza un documento Word y resalta el texto que probablemente Turnitin marcaría como generado con IA.

## Instalación

```bash
git clone https://github.com/Eduardo0719382/Turnitin.git
```

Requiere Python 3.10 o superior. Ejecuta una vez `Instalar dependencias.bat` y luego abre `Detector IA.bat`.
La primera vez descarga un GPT-2 en español (~500 MB) de Hugging Face; después funciona sin conexión.

## Funciones

- **Analizar**: genera un reporte HTML con el porcentaje y las marcas. Si además eliges el PDF de Turnitin, muestra la comparación.
- **Aprender de este reporte**: con un Word y su reporte de Turnitin, memoriza las marcas exactas y reentrena el modelo. Lo aprendido se guarda en `datos/` y no se pierde al actualizar.
- **Buscar actualización**: consulta este repositorio y instala la última versión.

## Cómo decide

1. **Memoria de reportes**: si una oración ya fue evaluada por Turnitin en un reporte aprendido, se usa ese veredicto tal cual. Solo se guardan huellas (hashes), no el texto.
2. **Modelo estadístico**: para el texto nuevo, una regresión logística sobre perplejidad (GPT-2 español), ráfaga y rasgos de estilo en ventanas de oraciones. Es una estimación: en validación cruzada acierta bastante menos que sobre los documentos aprendidos.

## Consola

```bash
python detector.py analizar documento.docx --reporte turnitin.pdf
python detector.py aprender documento.docx turnitin.pdf
python detector.py actualizar
```
