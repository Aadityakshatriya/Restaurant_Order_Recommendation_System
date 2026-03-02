const state = {
  users: [],
  restaurants: [],
  selectedUser: "",
  selectedRestaurant: "",
  menuItems: [],
  combos: [],
  cart: [],
  itemById: new Map(),
  runToken: 0,
};

const els = {
  userSelect: document.getElementById("userSelect"),
  restaurantSelect: document.getElementById("restaurantSelect"),
  restaurantMeta: document.getElementById("restaurantMeta"),
  userCount: document.getElementById("userCount"),
  menuCount: document.getElementById("menuCount"),
  comboCount: document.getElementById("comboCount"),
  menuGrid: document.getElementById("menuGrid"),
  comboGrid: document.getElementById("comboGrid"),
  cartList: document.getElementById("cartList"),
  clearCartBtn: document.getElementById("clearCartBtn"),
  recState: document.getElementById("recState"),
  recList: document.getElementById("recList"),
  latencyBadge: document.getElementById("latencyBadge"),
};

function priceFmt(v) {
  const n = Number(v || 0);
  return `Rs ${n.toFixed(0)}`;
}

function chip(label) {
  return `<span class="meta-pill">${label}</span>`;
}

function setRecState(msg) {
  els.recState.textContent = msg;
}

function resetRecommendationState(message, invalidateInFlight = false) {
  if (invalidateInFlight) {
    state.runToken += 1;
  }
  els.recList.innerHTML = "";
  setRecState(message);
  els.latencyBadge.textContent = "-- ms";
  els.latencyBadge.classList.remove("good", "warn", "bad");
}

function setLatency(ms) {
  const rounded = Math.max(0, Math.round(ms));
  els.latencyBadge.textContent = `${rounded} ms`;
  els.latencyBadge.classList.remove("good", "warn", "bad");
  if (rounded <= 300) {
    els.latencyBadge.classList.add("good");
  } else if (rounded <= 450) {
    els.latencyBadge.classList.add("warn");
  } else {
    els.latencyBadge.classList.add("bad");
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const msg = payload.detail || `Request failed (${response.status})`;
    throw new Error(msg);
  }
  return payload;
}

function renderUserSelect() {
  els.userSelect.innerHTML = '<option value="">Select a user ID</option>';
  for (const id of state.users) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id;
    els.userSelect.appendChild(opt);
  }
  els.userCount.textContent = `${state.users.length} existing users loaded`;
}

function formatRestaurantLabel(r) {
  const cuisine = r.rest_cuisine || "mixed";
  const city = r.city || "unknown city";
  return `${r.restaurant_id} • ${cuisine} • ${city}`;
}

function renderRestaurantSelect() {
  els.restaurantSelect.innerHTML = '<option value="">Select a restaurant</option>';
  for (const r of state.restaurants) {
    const opt = document.createElement("option");
    opt.value = r.restaurant_id;
    opt.textContent = formatRestaurantLabel(r);
    els.restaurantSelect.appendChild(opt);
  }
}

function renderRestaurantMeta(restaurantId) {
  const r = state.restaurants.find((x) => x.restaurant_id === restaurantId);
  if (!r) {
    els.restaurantMeta.innerHTML = "";
    return;
  }
  els.restaurantMeta.innerHTML = [
    chip(`ID ${r.restaurant_id}`),
    chip(`City ${r.city || "N/A"}`),
    chip(`Cuisine ${r.rest_cuisine || "N/A"}`),
    chip(`Rating ${Number(r.rest_rating || 0).toFixed(2)}`),
    chip(`Price Tier ${r.rest_price_tier || "N/A"}`),
    chip(`Menu ${r.menu_size || 0} items`),
  ].join("");
}

function renderMenu() {
  els.menuCount.textContent = `${state.menuItems.length} items`;
  if (!state.menuItems.length) {
    els.menuGrid.innerHTML = '<p class="small-note">No items found for this restaurant.</p>';
    return;
  }

  els.menuGrid.innerHTML = state.menuItems
    .map((item) => {
      const veg = Number(item.candidate_is_veg || 0) === 1 ? "Veg" : "Non-Veg";
      return `
        <article class="menu-item">
          <p class="item-title">${item.candidate_name || item.candidate_item_id}</p>
          <div class="item-meta">
            ${chip(item.candidate_category || "unknown")}
            ${chip(item.candidate_cuisine_tag || "misc")}
            ${chip(veg)}
            ${chip(`${Number(item.candidate_calories || 0).toFixed(0)} cal`)}
          </div>
          <div class="item-actions">
            <span class="price">${priceFmt(item.candidate_price)}</span>
            <button class="add-btn" data-add-item="${item.candidate_item_id}">Add to cart</button>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderCombos() {
  els.comboCount.textContent = `${state.combos.length} combos`;
  if (!state.combos.length) {
    els.comboGrid.innerHTML = '<p class="small-note">No combo templates available.</p>';
    return;
  }
  els.comboGrid.innerHTML = state.combos
    .map(
      (combo) => `
      <article class="combo-card">
        <h4>${combo.label}</h4>
        <p class="combo-line">Est. ${priceFmt(combo.estimated_price)} • Score ${Number(combo.combo_score || 0).toFixed(2)}</p>
        <div class="combo-row">
          <span class="meta-pill">${(combo.categories || []).join(" + ")}</span>
          <button class="add-btn" data-add-combo="${(combo.item_ids || []).join(",")}">Add combo</button>
        </div>
      </article>
    `
    )
    .join("");
}

function renderCart() {
  if (!state.cart.length) {
    els.cartList.innerHTML = '<li class="small-note">Cart is empty.</li>';
    return;
  }
  els.cartList.innerHTML = state.cart
    .map((itemId) => {
      const item = state.itemById.get(itemId);
      return `
        <li class="cart-item">
          <div class="cart-top">
            <strong>${item?.candidate_name || itemId}</strong>
            <button class="remove-btn" data-remove-item="${itemId}">Remove</button>
          </div>
          <div class="small-note">${item?.candidate_category || "unknown"} • ${priceFmt(item?.candidate_price)}</div>
        </li>
      `;
    })
    .join("");
}

function renderRecommendations(ids) {
  const top2 = (ids || []).slice(0, 2);
  if (!top2.length) {
    els.recList.innerHTML = "";
    setRecState("No recommendations available for current cart.");
    return;
  }
  setRecState("Model output: top 2 only.");
  els.recList.innerHTML = top2
    .map((id, index) => {
      const item = state.itemById.get(id);
      const name = item?.candidate_name || id;
      const category = item?.candidate_category || "unknown";
      return `
        <li class="rec-item">
          <div class="rec-top">
            <strong>${name}</strong>
            <span class="rec-rank">Rank ${index + 1}</span>
          </div>
          <div class="small-note">${id} • ${category} • ${priceFmt(item?.candidate_price)}</div>
        </li>
      `;
    })
    .join("");
}

async function loadOptions() {
  const payload = await fetchJson("/ui/options");
  state.users = payload.users || [];
  state.restaurants = payload.restaurants || [];
  renderUserSelect();
  renderRestaurantSelect();
}

async function loadRestaurantMenu(restaurantId) {
  const payload = await fetchJson(`/ui/restaurants/${encodeURIComponent(restaurantId)}/menu`);
  state.menuItems = payload.items || [];
  state.combos = payload.combos || [];
  state.itemById = new Map(state.menuItems.map((x) => [x.candidate_item_id, x]));
  state.cart = [];
  renderRestaurantMeta(restaurantId);
  renderMenu();
  renderCombos();
  renderCart();
  resetRecommendationState("Add an item to start recommendations.", true);
}

function addItemToCart(itemId, runModel = true) {
  if (!state.selectedUser || !state.selectedRestaurant) {
    return;
  }
  if (!state.itemById.has(itemId)) {
    return;
  }
  if (!state.cart.includes(itemId)) {
    state.cart.push(itemId);
    renderCart();
  }
  if (runModel) {
    runRecommendations();
  }
}

function addComboToCart(itemIds) {
  let changed = false;
  for (const id of itemIds) {
    if (state.itemById.has(id) && !state.cart.includes(id)) {
      state.cart.push(id);
      changed = true;
    }
  }
  if (changed) {
    renderCart();
    runRecommendations();
  }
}

async function runRecommendations() {
  if (!state.selectedUser || !state.selectedRestaurant) {
    return;
  }
  if (!state.cart.length) {
    resetRecommendationState("Add an item to start recommendations.", true);
    return;
  }

  const token = ++state.runToken;
  setRecState("Running inference...");
  const started = performance.now();

  try {
    const recommendResp = await fetchJson("/recommend-main", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: state.selectedUser,
        restaurant_id: state.selectedRestaurant,
        cart_item_ids: state.cart,
        request_id: `ui-${Date.now()}`,
        top_k: 2,
      }),
    });
    if (token !== state.runToken) return;

    const latency = performance.now() - started;
    setLatency(latency);
    renderRecommendations(recommendResp.recommended_item_ids || []);
  } catch (err) {
    if (token !== state.runToken) return;
    setRecState(err.message || "Inference request failed.");
    els.recList.innerHTML = "";
    setLatency(999);
  }
}

function removeFromCart(itemId) {
  const next = state.cart.filter((x) => x !== itemId);
  if (next.length === state.cart.length) return;
  state.cart = next;
  renderCart();
  runRecommendations();
}

function bindEvents() {
  els.userSelect.addEventListener("change", () => {
    state.selectedUser = els.userSelect.value || "";
    const enableRestaurant = Boolean(state.selectedUser);
    els.restaurantSelect.disabled = !enableRestaurant;
    if (!enableRestaurant) {
      state.selectedRestaurant = "";
      state.menuItems = [];
      state.combos = [];
      state.cart = [];
      state.itemById = new Map();
      renderRestaurantMeta("");
      renderMenu();
      renderCombos();
      renderCart();
      resetRecommendationState("Select a user and restaurant first.", true);
    }
  });

  els.restaurantSelect.addEventListener("change", async () => {
    state.selectedRestaurant = els.restaurantSelect.value || "";
    if (!state.selectedRestaurant) return;
    await loadRestaurantMenu(state.selectedRestaurant);
  });

  els.menuGrid.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const id = target.getAttribute("data-add-item");
    if (id) {
      addItemToCart(id, true);
    }
  });

  els.comboGrid.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const ids = target.getAttribute("data-add-combo");
    if (!ids) return;
    addComboToCart(ids.split(",").filter(Boolean));
  });

  els.cartList.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const id = target.getAttribute("data-remove-item");
    if (id) {
      removeFromCart(id);
    }
  });

  els.clearCartBtn.addEventListener("click", () => {
    state.cart = [];
    renderCart();
    resetRecommendationState("Add an item to start recommendations.", true);
  });
}

async function init() {
  setRecState("Loading options...");
  bindEvents();
  try {
    await loadOptions();
    setRecState("Select user and restaurant to begin.");
  } catch (err) {
    setRecState(`Failed to load UI options: ${err.message || err}`);
  }
}

init();
