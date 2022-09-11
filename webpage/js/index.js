// ONNX function
async function create_session() {
    const session = await ort.InferenceSession.create('onnx/OMORI_SPRITE_AIM_25_LONG_GAN_epoch=9999_g_loss=2.7776_d_loss=0.2691.onnx')
    return session
}

async function generate(session, random_input) {
    // create Z
    const dims = [1, 16, 1, 1]
    const size = 16
    let inputData = NaN
    if (random_input.length == 0) {
        inputData = Float32Array.from({ length: size }, () => Math.random() * 2 - 1) // TODO: use random normal dist.
    } else {
        inputData = Float32Array.from(random_input)
    }
    const feeds = { "random_input": new ort.Tensor("float32", inputData, dims) };

    // generate image
    const results = await session.run(feeds);
    const img = results.image_output.data;

    // normalize and reshape
    let img_normalized = img.map((num) => Math.round((num + 1) / 2 * 255))
    let array_c = [] // C
    let array_h = [] // C,H
    let array_w = [] // C,H,W

    let ctr = 0
    for (let c = 0; c < 3; c++) {
        for (let h = 0; h < 32; h++) {
            for (let w = 0; w < 32; w++) {
                array_w[w] = img_normalized[ctr]
                ctr++
            }
            array_h[h] = array_w
            array_w = []
        }
        array_c[c] = array_h
        array_h = []
    }

    // create PNG buffer
    let buffer = new Uint8ClampedArray(32 * 32 * 4);

    ctr = 0
    for (let h = 0; h < 32; h++) {
        for (let w = 0; w < 32; w++) {
            buffer[ctr] = array_c[0][h][w]
            buffer[ctr + 1] = array_c[1][h][w]
            buffer[ctr + 2] = array_c[2][h][w]
            buffer[ctr + 3] = 255
            ctr += 4
        }
    }

    return buffer
}

async function arr_to_img(random_input, image_tag) {
    const img_promise = generate(session, random_input)

    var img_buffer = NaN
    img_promise.then(val => {
        img_buffer = val

        // create off-screen canvas element
        var canvas = document.createElement('canvas'),
            ctx = canvas.getContext('2d');

        canvas.width = 32;
        canvas.height = 32;

        // create imageData object
        var idata = ctx.createImageData(32, 32);

        // set our buffer as source
        idata.data.set(img_buffer);

        // update canvas with new data
        ctx.putImageData(idata, 0, 0);

        var dataUri = canvas.toDataURL()
        // to IMG
        let img = document.getElementById(image_tag)
        img.src = dataUri
    }).catch(err => {
        console.log(err);
    })
}

// Function for Tab 1
function create_random() {
    for (let row = 0; row < 5; row++) {
        for (let column = 0; column < 5; column++) {
            let sprite_id = 'sprite' + row + column
            arr_to_img([], sprite_id)
        }
    }
}

function download_all() {
    let zip = new JSZip()
    let ctr = 0
    for (let row = 0; row < 5; row++) {
        for (let column = 0; column < 5; column++) {
            let sprite_id = 'sprite' + row + column
            let sprite = document.getElementById(sprite_id)
            let sprite_uri = (sprite.src).split(',')[1]  // remove "data:image/png;base64" from uri

            zip.file(ctr++ + ".png", sprite_uri, { "base64": true })
        }
    }
    zip.generateAsync({ type: "blob" })
        .then(function (content) {
            saveAs(content, "25_sprite.zip");
        });
}

// Function for Tab 2
function create_manual() {
    let random_input = []
    for (let i = 0; i < 16; i++) {
        random_input[i] = Number(document.getElementById('d' + i).value) / 100
        document.getElementById('l' + i).innerText = random_input[i]
    }
    arr_to_img(random_input, 'sprite_manual')
}

async function randomize() {
    for (let s = 0; s < 16; s++) {
        let slider_id = 'd' + s
        let slider = document.getElementById(slider_id)
        slider.value = ((Math.random() * 2 - 1) * 100).toFixed(2)
    }
    create_manual()
}

// Initialization

// Tabs
const tabsContainer = document.querySelector(".tabs")
var instance = M.Tabs.init(tabsContainer, { 'duration': 0 });

// Collapsible/FAQ tab
document.addEventListener('DOMContentLoaded', function () {
    var elems = document.querySelectorAll('.collapsible');
    var instances = M.Collapsible.init(elems, {
        // specify options here
    });
});

// ONNX session
const session_promise = create_session()
let session = NaN
session_promise.then(val => {
    session = val
})
