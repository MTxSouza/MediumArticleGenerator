function get_input_text() {
    var title = document.getElementById("title").value;
    return title;
}

function show_error_message(error, message) {
    // Get the generation div element
    const genDiv = document.getElementById("generation");

    // Create a transparent red box element
    const errorBox = document.createElement("div");
    errorBox.style.backgroundColor = "rgba(255, 0, 0, 0.25)";
    errorBox.style.margin = "auto";
    errorBox.style.width = "400px"
    errorBox.style.padding = "20px";
    errorBox.style.borderRadius = "10px";
    errorBox.style.color = "red";
    errorBox.style.fontFamily = "serif";
    errorBox.textContent = `Error: ${message}`;

    // Add the error box to the generation div
    genDiv.appendChild(errorBox);

    // Remove the error box after 3 seconds
    setTimeout(() => {
        genDiv.removeChild(errorBox);
    }, 3000);
    console.error("Error fetching data:", error);
}

async function request_article() {
    var title = get_input_text();

    // Send a request to the server
    try {
        const response = await fetch(`http://0.0.0.0:8000/generate?text=${title}`);
    } catch (error) {
        if (error instanceof TypeError) {
            show_error_message(error, "Could not connect to the server");
        } else {
            show_error_message(error, "Internal server error");
        }
        return;
    }

    // Check if the request was successful
    if (!response.ok) {
        const error = await response.text();
        show_error_message(error, "Failed to fetch data");
        return;
    }

    // Get the response from the server
    const result = await response.text();

    // Display the response
    document.getElementById("output").textContent = result;
}
