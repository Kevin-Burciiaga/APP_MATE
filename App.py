import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sympy as sp
import webbrowser

# Método de Euler mejorado (Heun)
def euler_mejorado(f, x0, y0, h, x_final):
    print("\nMétodo de Euler Mejorado:")
    x = x0
    y = y0
    x_vals = [x]
    y_vals = [y]
    f_vals = []
    hf_vals = []
    u_vals = []
    f_u_vals = []
    y_nuevo_vals = []
    n = int((x_final - x0) / h)
    for i in range(n):
        k1 = f(x, y)
        k2 = f(x + h, y + h * k1)
        y_nuevo = y + h * (k1 + k2) / 2
        x_nuevo = x + h
        x_vals.append(x_nuevo)
        y_vals.append(y_nuevo)
        f_vals.append(k1)
        hf_vals.append(h * k1)
        u_vals.append(y + h * k1)
        f_u_vals.append(k2)
        y_nuevo_vals.append(y_nuevo)
        print(f"Iteración {i+1}: x = {x:.4f}, y = {y:.4f}, f(x, y) = {k1:.4f}, h*f(x, y) = {h*k1:.4f}, U_{i+1} = {y + h * k1:.4f}, f(X_{i+1}, U_{i+1}) = {k2:.4f}, y_{i+1} = {y_nuevo:.4f}")
        if abs(y_nuevo) > 1e10:  # Controlar valores demasiado grandes
            print("Advertencia: Los valores se están volviendo demasiado grandes. Deteniendo la iteración.")
            break
        x = x_nuevo
        y = y_nuevo
    return x_vals, y_vals, f_vals, hf_vals, u_vals, f_u_vals, y_nuevo_vals

# Método de Newton-Raphson
def newton_raphson(f_str, x0, tol, max_iter):
    print("\nMétodo de Newton-Raphson:")
    x = sp.symbols('x')
    f = sp.sympify(f_str)
    df = sp.diff(f, x)
    f_lambdified = sp.lambdify(x, f, 'math')
    df_lambdified = sp.lambdify(x, df, 'math')
    
    if df_lambdified(x0) == 0:
        print("Derivada cero en el punto inicial. No se puede continuar.")
        return None
    
    x_vals = [x0]
    y_vals = [f_lambdified(x0)]
    for i in range(max_iter):
        fx = f_lambdified(x0)
        dfx = df_lambdified(x0)
        if dfx == 0:
            print("Derivada cero. No se puede continuar.")
            return None
        x_nuevo = x0 - fx / dfx
        x_vals.append(x_nuevo)
        y_vals.append(f_lambdified(x_nuevo))
        print(f"Iteración {i+1}: x = {x_nuevo:.4f}, f(x) = {fx:.4f}")
        if round(x_nuevo, tol) == round(x0, tol):
            print(f"Convergencia alcanzada en x = {x_nuevo:.{tol}f}")
            return x_vals, y_vals
        x0 = x_nuevo
    print("Máximo número de iteraciones alcanzado.")
    return x_vals, y_vals

# Método de Runge-Kutta de cuarto orden
def runge_kutta_4to_orden(f, x0, y0, h, x_final):
    print("\nMétodo de Runge-Kutta (4to orden):")
    x = x0
    y = y0
    x_vals = [x]
    y_vals = [y]
    f_vals = []
    hf_vals = []
    u_vals = []
    f_u_vals = []
    y_nuevo_vals = []
    n = int((x_final - x0) / h)
    for i in range(n):
        k1 = f(x, y)
        k2 = f(x + h/2, y + (h/2) * k1)
        k3 = f(x + h/2, y + (h/2) * k2)
        k4 = f(x + h, y + h * k3)
        y_nuevo = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        x_nuevo = x + h
        x_vals.append(x_nuevo)
        y_vals.append(y_nuevo)
        f_vals.append(k1)
        hf_vals.append(h * k1)
        u_vals.append(y + h * k1)
        f_u_vals.append(k2)
        y_nuevo_vals.append(y_nuevo)
        print(f"Iteración {i+1}: x = {x:.4f}, y = {y:.4f}, f(x, y) = {k1:.4f}, h*f(x, y) = {h*k1:.4f}, U_{i+1} = {y + h * k1:.4f}, f(X_{i+1}, U_{i+1}) = {k2:.4f}, y_{i+1} = {y_nuevo:.4f}")
        if abs(y_nuevo) > 1e10:  # Controlar valores demasiado grandes
            print("Advertencia: Los valores se están volviendo demasiado grandes. Deteniendo la iteración.")
            break
        x = x_nuevo
        y = y_nuevo
    return x_vals, y_vals, f_vals, hf_vals, u_vals, f_u_vals, y_nuevo_vals

# Función para ingresar una función desde la consola
def ingresar_funcion(metodo):
    while True:
        try:
            if metodo == "Newton-Raphson":
                print("Ingresa la función en términos de 'x'.")
                print("Ejemplo para Newton-Raphson: 'x**2 - 2'")
            else:
                print("Ingresa la función en términos de 'x' y 'y'.")
                print("Ejemplo para Euler mejorado o Runge-Kutta: 'x * y'")
            funcion_str = input("Función: ")
            # Validar la función
            test_x = 1.0
            test_y = 1.0 if 'y' in funcion_str else None
            if metodo == "Newton-Raphson":
                eval(funcion_str, {'x': test_x, 'math': math})
            else:
                if test_y:
                    eval(funcion_str, {'x': test_x, 'y': test_y, 'math': math})
                else:
                    eval(funcion_str, {'x': test_x, 'math': math})
            return lambda x, y=None: eval(funcion_str, {'x': x, 'y': y, 'math': math}), funcion_str
        except Exception as e:
            print(f"Error: La función ingresada no es válida. Detalles: {e}")

# Función para validar entradas numéricas
def validar_entrada_numerica(mensaje, tipo=float, positivo=True):
    while True:
        try:
            valor = tipo(input(mensaje))
            if positivo and valor < 0:
                print("Error: Debes ingresar un número positivo.")
                continue
            return valor
        except ValueError:
            print("Error: Debes ingresar un número válido.")

# Función para graficar y exportar a HTML
def graficar_y_exportar(x_vals, y_vals, f_vals, hf_vals, u_vals, f_u_vals, y_nuevo_vals, titulo, metodo):
    # Exportar a HTML con plotly
    fig = go.Figure()

    if metodo == "Newton-Raphson":
        # Añadir la tabla de iteraciones solo con x y y
        fig.add_trace(go.Table(
            header=dict(values=['Iteración', 'x', 'y']),
            cells=dict(values=[list(range(len(x_vals))), x_vals, y_vals])
        ))
    else:
        # Añadir la tabla de iteraciones con todos los valores
        fig.add_trace(go.Table(
            header=dict(values=['Iteración', 'x', 'y', 'f(x, y)', 'h*f(x, y)', 'U_{i+1}', 'f(X_{i+1}, U_{i+1})', 'y_{i+1}']),
            cells=dict(values=[list(range(len(x_vals))), x_vals, y_vals, f_vals, hf_vals, u_vals, f_u_vals, y_nuevo_vals])
        ))

    # Actualizar el diseño del gráfico
    fig.update_layout(
        title=titulo,
        height=800  # Ajustar la altura para acomodar la tabla
    )

    # Guardar el archivo HTML
    nombre_archivo = f"grafica_{metodo.lower().replace(' ', '_')}.html"
    fig.write_html(nombre_archivo)
    print(f"Tabla exportada como '{nombre_archivo}'")

    # Abrir el archivo HTML automáticamente
    webbrowser.open(nombre_archivo)

# Menú principal
def menu():
    print("\n--- Menú de Métodos Numéricos ---")
    print("1. Método de Euler Mejorado")
    print("2. Método de Newton-Raphson")
    print("3. Método de Runge-Kutta (4to orden)")
    print("4. Ver instrucciones")
    print("5. Salir")
    opcion = input("Selecciona una opción (1-5): ")
    return opcion

# Instrucciones para el usuario
def mostrar_instrucciones():
    print("\n--- Instrucciones ---")
    print("1. Selecciona el método numérico que deseas utilizar:")
    print("   - 1: Método de Euler Mejorado")
    print("   - 2: Método de Newton-Raphson")
    print("   - 3: Método de Runge-Kutta (4to orden)")
    print("   - 4: Ver instrucciones")
    print("   - 5: Salir")
    print("2. Ingresa la función que deseas resolver:")
    print("   - Para Euler mejorado y Runge-Kutta, la función debe ser en términos de 'x' y 'y' (por ejemplo, 'x * y').")
    print("   - Para Newton-Raphson, la función debe ser en términos de 'x' (por ejemplo, 'x**2 - 2').")
    print("3. Proporciona los parámetros necesarios para el método seleccionado:")
    print("   - Para Euler Mejorado y Runge-Kutta:")
    print("     - Valor inicial de x (x0): Número real, por ejemplo, 0.0")
    print("     - Valor inicial de y (y0): Número real, por ejemplo, 1.0")
    print("     - Tamaño del paso (h): Número real positivo, por ejemplo, 0.1")
    print("     - Valor final de x (x_final): Número real, por ejemplo, 1.0")
    print("   - Para Newton-Raphson:")
    print("     - Valor inicial de x (x0): Número real, por ejemplo, 1.0")
    print("     - Tolerancia (tol): Número entero positivo que indica la cantidad de dígitos después del punto decimal que deben ser iguales, por ejemplo, 6")
    print("     - Número máximo de iteraciones: Número entero positivo, por ejemplo, 100")
    print("4. Observa los resultados en la consola y la gráfica generada.")
    print("   - Los resultados numéricos se mostrarán en la consola.")
    print("   - La gráfica se mostrará en una ventana y se exportará como un archivo HTML.")
    print("   - La tabla de iteraciones se exportará como un archivo HTML.")

# Programa principal
if __name__ == "__main__":
    while True:
        opcion = menu()

        if opcion == "1":
            # Solicitar parámetros para Euler mejorado
            print("\nIngresa los parámetros para el método de Euler Mejorado:")
            print("Valor inicial de x (x0): Número real, por ejemplo, 0.0")
            print("El valor inicial de x (x0) es el punto de partida en el eje x.")
            x0 = validar_entrada_numerica("Valor inicial de x (x0): ", positivo=False)
            print("Valor inicial de y (y0): Número real, por ejemplo, 1.0")
            print("El valor inicial de y (y0) es el punto de partida en el eje y.")
            y0 = validar_entrada_numerica("Valor inicial de y (y0): ", positivo=False)
            print("Tamaño del paso (h): Número real positivo, por ejemplo, 0.1")
            print("El tamaño del paso (h) determina la distancia entre los puntos de la iteración.")
            h = validar_entrada_numerica("Tamaño del paso (h): ")
            print("Valor final de x (x_final): Número real, por ejemplo, 1.0")
            print("El valor final de x (x_final) es el punto en el eje x donde se detendrá la iteración.")
            x_final = validar_entrada_numerica("Valor final de x (x_final): ")
            f, _ = ingresar_funcion("Euler Mejorado")
            x_vals, y_vals, f_vals, hf_vals, u_vals, f_u_vals, y_nuevo_vals = euler_mejorado(f, x0, y0, h, x_final)
            graficar_y_exportar(x_vals, y_vals, f_vals, hf_vals, u_vals, f_u_vals, y_nuevo_vals, "Método de Euler Mejorado", "Euler Mejorado")

        elif opcion == "2":
            # Solicitar parámetros para Newton-Raphson
            print("\nIngresa los parámetros para el método de Newton-Raphson:")
            print("Valor inicial de x (x0): Número real, por ejemplo, 1.0")
            print("El valor inicial de x (x0) es el punto de partida en el eje x.")
            x0 = validar_entrada_numerica("Valor inicial de x (x0): ", positivo=False)
            print("Tolerancia (tol): Número entero positivo que indica la cantidad de dígitos después del punto decimal que deben ser iguales, por ejemplo, 6")
            tol = validar_entrada_numerica("Tolerancia (tol): ", int)
            print("Número máximo de iteraciones: Número entero positivo, por ejemplo, 100")
            print("El número máximo de iteraciones es el límite de iteraciones permitidas para encontrar la solución.")
            max_iter = validar_entrada_numerica("Número máximo de iteraciones: ", int)
            _, f_str = ingresar_funcion("Newton-Raphson")
            result = newton_raphson(f_str, x0, tol, max_iter)
            if result is not None:
                x_vals, y_vals = result
                graficar_y_exportar(x_vals, y_vals, [], [], [], [], [], "Método de Newton-Raphson", "Newton-Raphson")
            else:
                print("El método de Newton-Raphson no pudo converger.")

        elif opcion == "3":
            # Solicitar parámetros para Runge-Kutta de cuarto orden
            print("\nIngresa los parámetros para el método de Runge-Kutta (4to orden):")
            print("Valor inicial de x (x0): Número real, por ejemplo, 0.0")
            print("El valor inicial de x (x0) es el punto de partida en el eje x.")
            x0 = validar_entrada_numerica("Valor inicial de x (x0): ", positivo=False)
            print("Valor inicial de y (y0): Número real, por ejemplo, 1.0")
            print("El valor inicial de y (y0) es el punto de partida en el eje y.")
            y0 = validar_entrada_numerica("Valor inicial de y (y0): ", positivo=False)
            print("Tamaño del paso (h): Número real positivo, por ejemplo, 0.1")
            print("El tamaño del paso (h) determina la distancia entre los puntos de la iteración.")
            h = validar_entrada_numerica("Tamaño del paso (h): ")
            print("Valor final de x (x_final): Número real, por ejemplo, 1.0")
            print("El valor final de x (x_final) es el punto en el eje x donde se detendrá la iteración.")
            x_final = validar_entrada_numerica("Valor final de x (x_final): ")
            f, _ = ingresar_funcion("Runge-Kutta")
            x_vals, y_vals, f_vals, hf_vals, u_vals, f_u_vals, y_nuevo_vals = runge_kutta_4to_orden(f, x0, y0, h, x_final)
            graficar_y_exportar(x_vals, y_vals, f_vals, hf_vals, u_vals, f_u_vals, y_nuevo_vals, "Método de Runge-Kutta (4to orden)", "Runge-Kutta 4to Orden")

        elif opcion == "4":
            mostrar_instrucciones()

        elif opcion == "5":
            print("Saliendo del programa...")
            break

        else:
            print("Opción no válida. Intenta de nuevo.")