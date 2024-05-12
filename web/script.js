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
    genDiv.insertAdjacentElement("afterbegin", errorBox);

    // Remove the error box after 3 seconds
    setTimeout(() => {
        genDiv.removeChild(errorBox);
    }, 3000);
    console.error("Error fetching data:", error);
}

async function request_article() {
    var title = get_input_text();

    // Disable the button and input field
    button = document.getElementById("generate");
    button.disabled = true;
    input = document.getElementById("title");
    input.disabled = true;

    // Send a request to the server
    try {
        const response = await fetch("http://0.0.0.0:8000/generate", {
            method: "POST",
            headers: {
                "content-type": "application/json; charset=utf-8"
            },
            body: JSON.stringify({
                text : title,
                extra_tokens : 50,
                max_length : 80
            }),
        });

        // Check if the request was successful
        if (response.status == 400) {
            const error = await response.json();
            show_error_message(response.detail, error.detail);
            return;
        }

        // Get element to display the generation
        const genDiv = document.getElementById("generation");
        const genText = document.createElement("p");

        // Clear the generation div
        if (genDiv.childElementCount > 0) {
            genDiv.removeChild(genDiv.firstChild);
        }
        genDiv.appendChild(genText);

        // Get the response from the server
        const reader = response.body.getReader();
        let done, value;
        while (!done) {
            ({done, value} = await reader.read());
            if (done) {
                break;
            } else {
                const text = new TextDecoder("utf-8").decode(value);
                genText.textContent = genText.innerText + text;
            }
        }
    } catch (error) {
        if (error instanceof TypeError) {
            show_error_message(error, "Could not connect to the server");
        } else {
            show_error_message(error, "Internal server error");
        }
        return;
    } finally {
        // Enable the button and input field
        button.disabled = false;
        input.disabled = false;
    }
}
