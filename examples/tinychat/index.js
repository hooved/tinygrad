window.TINYCHAT_ROOT = "/";
window.MODEL_BASE_URL= ".";
//window.BACKEND = "WebGPU";
window.BACKEND = "WASM";

const tiktokenReady = (async () => {
  const { init, get_encoding, Tiktoken, load } = await import('./tiktoken.js');
  window.Tiktoken = Tiktoken;
  window.tiktokenInit = init;
  window.tiktokenGetEncoding = get_encoding;
  window.tiktokenLoad = load;
})();

const kernelsReady = (async () => {
  const exports = await import(`./net_clang.js?version=${Date.now()}`); // TODO: is cache-busting necessary
  Object.assign(self, exports);
})();

// copied from examples/webgpu/stable_diffusion/index.html
const getDevice = async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const requiredLimits = {};
  const maxBufferSizeInSDModel = 1073741824;
  requiredLimits.maxStorageBufferBindingSize = maxBufferSizeInSDModel;
  requiredLimits.maxBufferSize = maxBufferSizeInSDModel;
            
  return await adapter.requestDevice({
    requiredLimits
  });
};

// copied from examples/webgpu/stable_diffusion/index.html 
function initDb() {
  return new Promise((resolve, reject) => {
    let db;
    const request = indexedDB.open('tinydb', 1);
    request.onerror = (event) => {
      console.error('Database error:', event.target.error);
      resolve(null);
    };

    request.onsuccess = (event) => {
      db = event.target.result;
      console.log("Db initialized.");
      resolve(db);
    };

    request.onupgradeneeded = (event) => {
      db = event.target.result;
      if (!db.objectStoreNames.contains('tensors')) {
        db.createObjectStore('tensors', { keyPath: 'id' });
      }
    };
  });
}

// copied from examples/webgpu/stable_diffusion/index.html 
function readTensorFromDb(db, id) {
  return new Promise((resolve, reject) => {
    if (db == null) {
      resolve(null);
    }
            
      const transaction = db.transaction(['tensors'], 'readonly');
      const store = transaction.objectStore('tensors');
      const request = store.get(id);

      transaction.onabort = (event) => {
        console.log("Transaction error while reading tensor: " + event.target.error);
        resolve(null);
      };

      request.onsuccess = (event) => {
        const result = event.target.result;
        if (result) {
          resolve(result);
        } else {
          resolve(null);
        }
      };

      request.onerror = (event) => {
        console.error('Tensor retrieve failed: ', event.target.error);
        resolve(null);
      };
  });
}

function getAllKeysFromDb(db) {
  return new Promise((resolve, reject) => {
    if (db == null) {resolve([]);}
    const transaction = db.transaction(['tensors'], 'readonly');
    const store = transaction.objectStore('tensors');
    const request = store.getAllKeys();
    transaction.onabort = (event) => {
      console.log("Transaction error while reading IndexedDB keys: " + event.target.error);
      resolve([]);
    };
    request.onsuccess = function (event) {resolve(event.target.result);};
    request.onerror = (event) => {
      console.error('Retrieval of IndexedDB keys failed: ', event.target.error);
      resolve([]);
    };
  });
}

// modified from examples/webgpu/stable_diffusion/index.html 
function saveTensorToDb(db, id, tensor) {
  return readTensorFromDb(db, id).then((result) => {
    if (!result) {
      new Promise((resolve, reject) => {
        if (db == null) {
          resolve(null);
        }

        const transaction = db.transaction(['tensors'], 'readwrite');
        const store = transaction.objectStore('tensors');
        const request = store.put({ id: id, content: tensor });

        transaction.onabort = (event) => {
          console.log("Transaction error while saving tensor: " + event.target.error);
          resolve(null);
        };

        request.onsuccess = () => {
          console.log('Tensor saved successfully.');
          resolve();
        };

        request.onerror = (event) => {
          console.error('Tensor save failed:', event.target.error);
          resolve(null);
        };
      });
    } else {
      return null;
    }
  }).catch(()=> null);
}

function deleteTensorFromDb(db, id) {
  return new Promise((resolve, reject) => {
    if (db == null) {
      console.error("Database is not initialized.");
      resolve(null);
      return;
    }

    const transaction = db.transaction(['tensors'], 'readwrite');
    const store = transaction.objectStore('tensors');
    const request = store.delete(id);

    transaction.oncomplete = () => {
      console.log(`Tensor with ID '${id}' deleted successfully.`);
      resolve();
    };

    transaction.onerror = (event) => {
      console.error("Transaction error while deleting tensor:", event.target.error);
      resolve(null);
    };

    request.onerror = (event) => {
      console.error('Tensor deletion failed:', event.target.error);
      resolve(null);
    };

    request.onsuccess = () => {
      console.log(`Delete request for tensor with ID '${id}' succeeded.`);
    };
  });
}

async function hashBuffer(bytes) {
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, '0')).join('');
}

async function getFreePipeline(pipelinePool) {
  for (;;) {
    const idx = pipelinePool.findIndex(obj => !obj.busy);
    if (idx >= 0) {
      pipelinePool[idx].busy = true;
      return pipelinePool[idx].pipeline;
    }
    await new Promise(r => setTimeout(r, 5));
  }
}

function releasePipeline(pipeline, pipelinePool) {
  const obj = pipelinePool.find(obj => obj.pipeline === pipeline);
  if (obj) obj.busy = false;
}

function sendMessageToWorker(worker, message) {
  return new Promise((resolve, reject) => {
    const onMessage = (event) => {
      resolve(event.data);
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
    };

    const onError = (error) => {
      reject(error);
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
    };

    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', onError);

    if (message.header === "token") {worker.postMessage(message.data);}
    else if (message.header === "setup") {worker.postMessage(message.data);}
    // if message.data is a [k, v] from Object.entries(state_dict)
    else if (message.header === "k_v") {worker.postMessage(message.data, [message.data[1].bytes.buffer]);}
    // if message.data is the decompressed state_dict
    else if (message.header === "state_dict") {worker.postMessage(message.data, Object.values(message.data).map(({ bytes }) => bytes.buffer));}
  });
}

async function decompress(compressedBuffers, state_dict, device, progress) {
  let totalLoaded = 0;
  let totalSize = Object.values(state_dict).filter(item => item.dtype === "Q6_K").reduce((sum, item) => sum + item.size, 0);
  const numCheckpoints = 90;
  let nextCheckpoint = totalSize / numCheckpoints;
  const decompProgressFraction = 0.90;
  totalSize = totalSize / decompProgressFraction; // extend progress bar for minor steps after decompression
  const t0 = performance.now();

  const inChunkSize = (window.BACKEND === "WebGPU") ? 3144960 : 430080; // WebGPU value is max size that tinygrad compiled without exceptions; divisible by 210
  let inChunk = new Uint8Array(inChunkSize);
  const byteFactor = 1 / 210 * 256 * 4;
  let chunkContents = {};

  if (window.BACKEND === "WebGPU") {
    const num_decomposers = 8;
    // decompression time goes from 15sec to 10sec by scheduling GPU jobs like below
    // TODO: can we get tinygrad to give us bigger kernels? currently throws exceptions when trying to compile them
    const pipelinePool = await Promise.all(
      Array
        .from({ length: num_decomposers }, () => q6k_to_f32().setup(device))
        .map(async (promise) => {
          return {
            pipeline: await promise,
            busy: false
          };
        })
    );

    const dequantize = async(inChunk, chunkContents, decomp) => {
      let outChunk = await decomp(inChunk);
      outChunk = new Uint8Array(outChunk.buffer);
      for (const [t, start_end_tOffset] of Object.entries(chunkContents)) {
        const start = parseInt(start_end_tOffset[0] * byteFactor);
        const end = parseInt(start_end_tOffset[1] * byteFactor);
        const offset = parseInt(start_end_tOffset[2] * byteFactor);
        state_dict[t].bytes.set(outChunk.subarray(start, end), offset)
      }
      totalLoaded += inChunkSize;
      if (totalLoaded >= nextCheckpoint) {
        nextCheckpoint += totalSize * decompProgressFraction / numCheckpoints;
        progress(totalLoaded, totalSize, "Decompressing model:");
      }
    }

    function scheduleDequantizeJob() {
      const reserved_inChunk = inChunk;
      const reserved_chunkContents = chunkContents;
      freeSpace = inChunkSize;
      inChunk = new Uint8Array(inChunkSize);
      chunkContents = {};
      return (async () => {
        const d = await getFreePipeline(pipelinePool);
        await dequantize(reserved_inChunk, reserved_chunkContents, d);
        releasePipeline(d, pipelinePool);
      })();
    }

    const gpuJobs = [];
    let freeSpace = inChunkSize;

    // TODO: minimize memory overhead, consume each compressed buffer from end to end
    for (const [k, v] of Object.entries(state_dict)) {
      const tensor = compressedBuffers[v.chunk].subarray(v.start_pos, v.start_pos + v.size);

      if (v.dtype === "Q6_K") {
        v.size = parseInt(v.size * byteFactor);
        v.dtype = "float32";
        v.bytes = new Uint8Array(v.size);

        let tensor_cursor = 0;
        while (tensor_cursor < tensor.byteLength) {
          const inChunk_cursor = inChunkSize - freeSpace;
          if (!(k in chunkContents)) {chunkContents[k] = [inChunk_cursor, inChunk_cursor, tensor_cursor]}

          const end = Math.min(tensor_cursor + freeSpace, tensor.byteLength);
          inChunk.set(tensor.subarray(tensor_cursor, end), inChunk_cursor);
          freeSpace -= (end - tensor_cursor);
          chunkContents[k][1] += (end - tensor_cursor);
          tensor_cursor = end;

          if (freeSpace === 0) {gpuJobs.push(scheduleDequantizeJob());}
        }
      } else {v.bytes = tensor;}
    }

    if (freeSpace < inChunkSize) {
      inChunk.set(new Uint8Array(freeSpace), inChunkSize - freeSpace); // pad last partial chunk with zeroes
      gpuJobs.push(scheduleDequantizeJob());
    }

    await Promise.all(gpuJobs);
  } 
  else if (window.BACKEND == "WASM") {
    delete state_dict["tok_embeddings.weight"]; // uses same data as output.weight

    const num_decomposers = navigator.hardwareConcurrency - 1;
    const workers = Array.from({ length: num_decomposers }, () => new Worker(`./worker.js?version=${Date.now()}`));
    const promises = workers.map(async (worker) => {
      await sendMessageToWorker(worker, {header: "setup", data: "decompress"}); // setup flag, worker can only do decompression now
      return {
        worker: worker,
        pipeline: (k_v_pair) => sendMessageToWorker(worker, {header: "k_v", data: k_v_pair}),
        busy: false
      };
    });
    const pipelinePool = await Promise.all(promises);

    function scheduleDequantizeJob(k, v) {
      // k, v are from the model's state_dict
      return (async () => {
        const pipeline = await getFreePipeline(pipelinePool);
        const new_v = await pipeline([k, v]);
        if (k.includes("feed_forward") || k.includes("attention.w")) {
          state_dict[k.replace("weight", "scale")] = {"dtype": "float32", "bytes": new_v.scale, "size": new_v.scale.length}
        }
        state_dict[k] = new_v;
        releasePipeline(pipeline, pipelinePool);
      })();
    }

    const cpuJobs = [];
    for (const [k, v] of Object.entries(state_dict)) {
      if (v.dtype === "Q6_K") {cpuJobs.push(scheduleDequantizeJob(k, v));}
    }
    await Promise.all(cpuJobs);
    pipelinePool.forEach(p => p.worker.terminate());

  } else {throw new Error(`window.BACKEND is ${window.BACKEND}, but must be WebGPU or WASM`)}

  const t1 = performance.now();
  console.log(`decompression elapsed seconds: ${(t1 - t0) / 1000}`)
}

const getAndDecompressGGUFChunks = async (device, progress) => {
  let totalLoaded = 0;
  let totalSize = 0;
  let partSize = {};

  const progressCallback = (part, loaded, total, message) => {
    totalLoaded += loaded;

    if (!partSize[part]) {
      totalSize += total;
      partSize[part] = true;
    }
                
    progress(totalLoaded, totalSize, message);
  };

  // modified from examples/webgpu/stable_diffusion/index.html getProgressDlForPart
  const loadPart = async (part, progressCallback) => {
      const response = await fetch(part);
      const contentLength = response.headers.get('content-length');
      const total = parseInt(contentLength, 10);

      const res = new Response(new ReadableStream({
          async start(controller) {
              const reader = response.body.getReader();
              for (;;) {
                  const { done, value } = await reader.read();
                  if (done) break;
                  progressCallback(part, value.byteLength, total, "Downloading model:");
                  controller.enqueue(value);
              }
                    
              controller.close();
          },
      }));
        
      return res.arrayBuffer();
  };

  const response = await fetch(`${window.MODEL_BASE_URL}/net_metadata.json`);
  // TODO: cache metadata
  const data = await response.json();
  const state_dict = data.metadata.state_dict;

  let db = await initDb();

  const getPart = async(filename, hash) => {
    let part = await readTensorFromDb(db, hash);

    if (part) {
      console.log(`Cache hit: ${filename}, hash: ${hash}`);
      totalLoaded += part.content.byteLength;
      totalSize += part.content.byteLength;
      progress(totalLoaded, totalSize, "Downloading model:")
      return Promise.resolve(part.content);
    } else {
      console.log(`Cache miss: ${filename}, hash: ${hash}`);
      return loadPart(`${window.MODEL_BASE_URL}/${filename}`, progressCallback);
    }
  }

  const correctHashes = data.metadata.chunks.map(chunk => chunk.hash)
  const compressedBuffers = await Promise.all(data.metadata.chunks.map(chunk => getPart(chunk.name, chunk.hash)));

  // delete unused cached buffers to free disk space -- if we update weights, user will otherwise have obsolete cached buffers
  const dbKeys = await getAllKeysFromDb(db);
  const correctHashesSet = new Set(correctHashes);
  const notInCorrectHashes = dbKeys.filter(key => !correctHashesSet.has(key));
  for (const hash of notInCorrectHashes) {deleteTensorFromDb(db, hash);}

  for (let i = 0; i < compressedBuffers.length; i++) {
    compressedBuffers[i] = new Uint8Array(compressedBuffers[i]);
    saveTensorToDb(db, correctHashes[i], compressedBuffers[i]);
  }

  // load state_dict with weights
  // split weights into one ArrayBuffer per weight, necessary for zero-copy hand-off of individual weights to separate workers
  // copy out weights from one parent buffer at a time, then delete the parent buffer, to minimize memory overhead
  const chunked_state_dict = Array.from({ length: compressedBuffers.length }, () => []);
  Object.entries(state_dict).forEach(([k, v]) => {
    chunked_state_dict[v.chunk].push([k, v]);
  });
  for (let i = 0; i < compressedBuffers.length; i++) {
    for (const [k, v] of chunked_state_dict[i]) {
      state_dict[k].bytes = compressedBuffers[i].slice(v.start_pos, v.start_pos + v.size);
    }
    compressedBuffers[i] = null;
  }

  await decompress(compressedBuffers, state_dict, device, progress);

  // TODO: refactor WebGPU backend to only use state_dict instead of decompressedBuffers; probably can get rid of below stuff
  if (window.BACKEND === "WebGPU") {
    // FFD bin packing
    const maxChunkSize = 1149173760; // byte size of float32 output.weight in llama-1B
    const chunks = [];
    const size_sorted_tensors = Object.entries(state_dict)
      .map(([key, value]) => ({name: key, size: value.size}))
      .sort((a, b) => b.size - a.size);
    for (const t of size_sorted_tensors) {
      let placed = false;
      for (const chunk of chunks) {
        const currentSum = chunk.reduce((sum, i) => sum + i.size, 0);
        if (currentSum + t.size <= maxChunkSize) {
          chunk.push(t);
          placed = true;
          break;
        }
      }
      if (!placed) chunks.push([t]);
    }
    progress(0.95, 1.0, "Decompressing model:");

    const decompressedBuffers = [];
    for (let i=0; i<chunks.length; i++) {
      const chunk = chunks[i];
      const chunkSize = chunk.reduce((sum, j) => sum + j.size, 0);
      decompressedBuffers.push(new Uint8Array(chunkSize));
      let cursor = 0;
      for (let j=0; j<chunk.length; j++) {
        const t = chunk[j];
        decompressedBuffers[i].set(state_dict[t.name].bytes, cursor);
        state_dict[t.name].bytes = null;
        state_dict[t.name].chunk = i;
        state_dict[t.name].start_pos = cursor;
        cursor += t.size;
        if (j % 5 === 0) await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
      }
    }

    progress(1.0, 1.0, "Decompressing model:");
    return {chunks: decompressedBuffers, metadata: state_dict};
  }
  else if (window.BACKEND === "WASM") {return {metadata: state_dict};}
};

document.addEventListener("alpine:init", () => {
  Alpine.data("state", () => ({
    // loadingMessage updates the user on page load progress, including weights download and decompression
    // if loadingMessage is not '', then prompt box will be hidden: this is default behavior on page load
    loadingMessage: 'Loading...',
    // model
    nets: {},
    tokenizer: null,
    // TODO: implement context sliding; model currently outputs gibberish past max_context
    max_context: 1024,
    lastSeenToks: [],

    progress(loaded, total, message) {
      const percentage = total ? Math.trunc((loaded / total) * 100) : 0;
      document.querySelector('.progress').style.width = `${percentage}%`;
      document.getElementById('progress-percentage').textContent = `${percentage}%`;
      if (message) {
        this.loadingMessage = message;
        document.getElementById('loading-message').textContent = this.loadingMessage;
      }
    },

    async init() {
      var device = null;
      if (window.BACKEND === "WebGPU") {
        try {
          device = await getDevice();
          console.log("WebGPU device initialized");
        } catch (error) {this.progress(0, 100, "Failed to launch WebGPU. Please check if WebGPU is enabled and reload the page. || Loading:"); console.log(error); return;}
      }

      try {
        var tensorData = await getAndDecompressGGUFChunks(device, this.progress.bind(this));
      } catch (error) {this.progress(0, 100, "Error decompressing model"); console.log(error); return;}

      var p = 0;
      try {
        this.progress(p, 100, "Loading tokenizer:");
        const wasmResponse = await fetch(`${window.MODEL_BASE_URL}/tiktoken_bg.wasm`);
        p = 10; this.progress(p, 100, "Loading tokenizer:");
        const wasmBytes = await wasmResponse.arrayBuffer();
        await tiktokenReady;
        await window.tiktokenInit((imports) => WebAssembly.instantiate(wasmBytes, imports));
        p = 20; this.progress(p, 100, "Loading tokenizer:");

        this.tokenizer = await createTokenizer(`${window.MODEL_BASE_URL}/llama3-2.tiktoken`);
        const tokenizer_works = (new TextDecoder().decode(this.tokenizer.decode(this.tokenizer.encode("hello world"))) === "hello world");
        console.log("tokenizer works:", tokenizer_works)
        p = 30; this.progress(p, 100, "Loading tokenizer:");
      } catch (error) {this.progress(p, 100, "Error launching tokenizer"); console.log(error); return;}

      try {
        p = 40; this.progress(p, 100, `Launching ${window.BACKEND} model:`);
        // TODO: clean this up
        if (window.BACKEND === "WebGPU") {
          let models = ["transformer"];
          const args = (window.BACKEND === "WebGPU") ? [device, tensorData.chunks, tensorData.metadata] : [tensorData.metadata];
          this.nets = await Promise.all([
                  //transformer().setup(device, tensorData.chunks, tensorData.metadata, this.progress.bind(this)),
                  transformer().setup(...[...args, this.progress.bind(this)]),
              ]).then((loadedModels) => loadedModels.reduce((acc, model, index) => { acc[models[index]] = model; return acc; }, {}))
        }
        else if (window.BACKEND === "WASM") {
          await kernelsReady;
          const modelWorker = new Worker(`./worker.js?version=${Date.now()}`);
          let msg = await sendMessageToWorker(modelWorker, {header: "setup", data: "setup_transformer"});
          msg = await sendMessageToWorker(modelWorker, {header: "state_dict", data: tensorData.metadata});
          this.nets = {"transformer": async (tok, start_pos) => sendMessageToWorker(modelWorker, {header: "token", data: [tok, start_pos]})};
        }
        this.progress(100, 100, `Launching ${window.BACKEND} model:`);
        this.loadingMessage = ""; // Triggers removal of loading bar, display of prompt box
      } catch (error) {this.progress(p, 100, "Error launching model"); console.log(error); return;}
    },

    // current state
    cstate: {
      time: null,
      messages: [],
    },

    // historical state
    histories: JSON.parse(localStorage.getItem("histories")) || [],

    home: 0,
    generating: false,
    endpoint: `${window.location.origin}/v1`,

    // performance tracking
    time_till_first: 0,
    tokens_per_second: 0,
    total_tokens: 0,

    removeHistory(cstate) {
      const index = this.histories.findIndex((state) => {
        return state.time === cstate.time;
      });
      if (index !== -1) {
        this.histories.splice(index, 1);
        localStorage.setItem("histories", JSON.stringify(this.histories));
      }
    },

    async handleSend() {
      const el = document.getElementById("input-form");
      const value = el.value.trim();
      if (!value) return;

      if (this.generating) return;
      // TODO: fix bug: if we switch to another chat session during generation, prompt bar locks up with "Generating..."
      this.generating = true;
      if (this.home === 0) this.home = 1;

      // ensure that going back in history will go back to home
      window.history.pushState({}, "", window.TINYCHAT_ROOT || "/");

      // add message to list
      this.cstate.messages.push({ role: "user", content: value });

      // clear textarea
      el.value = "";
      el.style.height = "auto";
      el.style.height = el.scrollHeight + "px";

      // reset performance tracking
      const prefill_start = Date.now();
      let start_time = 0;
      let tokens = 0;
      this.tokens_per_second = 0;

      // start receiving server sent events
      let gottenFirstChunk = false;
      for await (
        const chunk of this.openaiChatCompletion(this.cstate.messages)
      ) {
        if (!gottenFirstChunk) {
          this.cstate.messages.push({ role: "assistant", content: "" });
          gottenFirstChunk = true;
        }

        // add chunk to the last message
        // TODO: handle errors with localStorage overflow
        //   possible example: this.cstate.messages[...] was undefined when trying to prompt within an old cstate (chat session)
        this.cstate.messages[this.cstate.messages.length - 1].content += chunk;

        // calculate performance tracking
        tokens += 1;
        this.total_tokens += 1;
        if (start_time === 0) {
          start_time = Date.now();
          this.time_till_first = start_time - prefill_start;
        } else {
          const diff = Date.now() - start_time;
          if (diff > 0) {
            this.tokens_per_second = tokens / (diff / 1000);
          }
        }
      }

      // update the state in histories or add it if it doesn't exist
      const index = this.histories.findIndex((cstate) => {
        return cstate.time === this.cstate.time;
      });
      this.cstate.time = Date.now();
      if (index !== -1) {
        // update the time
        this.histories[index] = this.cstate;
      } else {
        this.histories.push(this.cstate);
      }
      // update in local storage
      localStorage.setItem("histories", JSON.stringify(this.histories));

      this.generating = false;
    },

    async handleEnter(event) {
      // if shift is not pressed
      if (!event.shiftKey) {
        event.preventDefault();
        await this.handleSend();
      }
    },

    updateTotalTokens(messages) {
      try {
        let toks = [this.tokenizer.bos_id];
        messages.forEach((message) => {
          if (!message.role || !message.content) {
            throw new Error("Each message must have a 'role' and 'content' property.");
          }
          toks = toks.concat(this.tokenizer.encodeMessage(message.role, message.content));

          if (messages.length > 0 && messages[messages.length - 1].role === "user") {
            toks = toks.concat(this.tokenizer.encodeRole("assistant"));
          }
          this.total_tokens = toks.length;
        });
      } catch (error) {
        console.error("Error updating total tokens:", error);
      }
    },

    async *openaiChatCompletion(messages) {
      let tokens = [this.tokenizer.bos_id];
      for (const message of messages) {
        tokens = tokens.concat(this.tokenizer.encodeMessage(message.role, message.content));
      }
      tokens = tokens.concat(this.tokenizer.encodeRole("assistant"));
      let startPos = 0
      let prefillToks = tokens.slice(0, -1);

      // Skip the largest possible sequence of tokens already represented at the beginning of the model's kv caches
      for (let i=0; i <= prefillToks.length; i++) {
        startPos = i;
        if (i == prefillToks.length) break;
        if (i == this.lastSeenToks.length) break;
        if (prefillToks[i] !== this.lastSeenToks[i]) break;
      }
      this.lastSeenToks = prefillToks;
      prefillToks = prefillToks.slice(startPos);

      for (const tok of prefillToks) {
        if (window.BACKEND === "WebGPU") {await this.nets["transformer"](new Float32Array([[tok]]), startPos);}
        else {await this.nets["transformer"](tok, startPos);}
        startPos += 1;
      }

      let lastTok = tokens[tokens.length - 1];
      while (true) {
        if (window.BACKEND === "WebGPU") {var tok = await this.nets["transformer"](new Float32Array([[lastTok]]), startPos); tok = tok[0];}
        else {var tok = await this.nets["transformer"](tok, startPos);}
        this.lastSeenToks.push(lastTok); // lets us skip prefilling with these tokens at the next prompt in this chain
        startPos += 1;
        lastTok = tok;
        if (this.tokenizer.stop_tokens.has(lastTok)) break;
        yield new TextDecoder().decode(this.tokenizer.decode([lastTok]));
      }
    },
  }));
});

const { markedHighlight } = globalThis.markedHighlight;
marked.use(markedHighlight({
  langPrefix: "hljs language-",
  highlight(code, lang, _info) {
    const language = hljs.getLanguage(lang) ? lang : "plaintext";
    return hljs.highlight(code, { language }).value;
  },
}));

// **** eventsource-parser ****
class EventSourceParserStream extends TransformStream {
  constructor() {
    let parser;

    super({
      start(controller) {
        parser = createParser((event) => {
          if (event.type === "event") {
            controller.enqueue(event);
          }
        });
      },

      transform(chunk) {
        parser.feed(chunk);
      },
    });
  }
}

function createParser(onParse) {
  let isFirstChunk;
  let buffer;
  let startingPosition;
  let startingFieldLength;
  let eventId;
  let eventName;
  let data;
  reset();
  return {
    feed,
    reset,
  };
  function reset() {
    isFirstChunk = true;
    buffer = "";
    startingPosition = 0;
    startingFieldLength = -1;
    eventId = void 0;
    eventName = void 0;
    data = "";
  }
  function feed(chunk) {
    buffer = buffer ? buffer + chunk : chunk;
    if (isFirstChunk && hasBom(buffer)) {
      buffer = buffer.slice(BOM.length);
    }
    isFirstChunk = false;
    const length = buffer.length;
    let position = 0;
    let discardTrailingNewline = false;
    while (position < length) {
      if (discardTrailingNewline) {
        if (buffer[position] === "\n") {
          ++position;
        }
        discardTrailingNewline = false;
      }
      let lineLength = -1;
      let fieldLength = startingFieldLength;
      let character;
      for (
        let index = startingPosition;
        lineLength < 0 && index < length;
        ++index
      ) {
        character = buffer[index];
        if (character === ":" && fieldLength < 0) {
          fieldLength = index - position;
        } else if (character === "\r") {
          discardTrailingNewline = true;
          lineLength = index - position;
        } else if (character === "\n") {
          lineLength = index - position;
        }
      }
      if (lineLength < 0) {
        startingPosition = length - position;
        startingFieldLength = fieldLength;
        break;
      } else {
        startingPosition = 0;
        startingFieldLength = -1;
      }
      parseEventStreamLine(buffer, position, fieldLength, lineLength);
      position += lineLength + 1;
    }
    if (position === length) {
      buffer = "";
    } else if (position > 0) {
      buffer = buffer.slice(position);
    }
  }
  function parseEventStreamLine(lineBuffer, index, fieldLength, lineLength) {
    if (lineLength === 0) {
      if (data.length > 0) {
        onParse({
          type: "event",
          id: eventId,
          event: eventName || void 0,
          data: data.slice(0, -1),
          // remove trailing newline
        });

        data = "";
        eventId = void 0;
      }
      eventName = void 0;
      return;
    }
    const noValue = fieldLength < 0;
    const field = lineBuffer.slice(
      index,
      index + (noValue ? lineLength : fieldLength),
    );
    let step = 0;
    if (noValue) {
      step = lineLength;
    } else if (lineBuffer[index + fieldLength + 1] === " ") {
      step = fieldLength + 2;
    } else {
      step = fieldLength + 1;
    }
    const position = index + step;
    const valueLength = lineLength - step;
    const value = lineBuffer.slice(position, position + valueLength).toString();
    if (field === "data") {
      data += value ? "".concat(value, "\n") : "\n";
    } else if (field === "event") {
      eventName = value;
    } else if (field === "id" && !value.includes("\0")) {
      eventId = value;
    } else if (field === "retry") {
      const retry = parseInt(value, 10);
      if (!Number.isNaN(retry)) {
        onParse({
          type: "reconnect-interval",
          value: retry,
        });
      }
    }
  }
}
const BOM = [239, 187, 191];
function hasBom(buffer) {
  return BOM.every((charCode, index) => buffer.charCodeAt(index) === charCode);
}

const PAT_STR = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

async function createTokenizer(bpeUrl) {
  const num_base_tokens = 128000;
  const special_tokens = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>": 128001,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>": 128007,
    "<|eot_id|>": 128009
  };
  const model = await window.tiktokenLoad({
        "load_tiktoken_bpe": bpeUrl,
        "special_tokens": special_tokens,
        "pat_str": PAT_STR
    });
  const tokenizer = new window.Tiktoken(model.bpe_ranks, model.special_tokens, model.pat_str)

  return {
    get bos_id() {
      return special_tokens["<|begin_of_text|>"];
    },

    get stop_tokens() {
      return new Set([
        special_tokens["<|end_of_text|>"],
        special_tokens["<|eot_id|>"],
      ]);
    },

    decode(toks) {
      const filtered = toks.filter((t) => t < num_base_tokens);
      return tokenizer.decode(filtered);
    },

    encode(text, allow_special = false) {
      const allowedSpecial = allow_special ? "all" : new Set();
      const disallowedSpecial = new Set();
      return tokenizer.encode(text, allowedSpecial, disallowedSpecial);
    },

    encodeRole(role) {
      const tokens = [];
      tokens.push(special_tokens["<|start_header_id|>"]);
      tokens.push(...this.encode(role));
      tokens.push(special_tokens["<|end_header_id|>"]);
      tokens.push(...this.encode("\n\n"));
      return tokens;
    },

    encodeMessage(role, content) {
      const roleTokens = this.encodeRole(role);
      const contentTokens = this.encode(content.trim());
      return [...roleTokens, ...contentTokens, special_tokens["<|eot_id|>"]];
    },
  };
}