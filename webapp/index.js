const resultEl = document.querySelector('.result pre');
const queryEl = document.getElementById('query');

const btn = document.querySelector('.search-bar button');
btn.addEventListener('click', function () {
    fetch('https://127.0.0.1:8080//publishers/publisher_1>/segments/query?q=' + queryEl.textContent)
        .then(function (result) {
            console.log(result);
        })
        .catch(function (err) {
            console.error(err);
        });
});