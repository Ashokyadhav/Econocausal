import { useEffect, useState } from 'react'
import './App.css'

import {
  getProducts,
  createProduct,
  updateProduct,
  deleteProduct,
} from './api/products'


const initialProducts = [
  {
    id: 1,
    name: 'Wireless Headphones',
    category: 'Electronics',
    price: 2499,
    demand: 'High',
    stock: 42,
  },
  {
    id: 2,
    name: 'Running Shoes',
    category: 'Fashion',
    price: 3299,
    demand: 'Medium',
    stock: 28,
  },
  {
    id: 3,
    name: 'Smart Watch',
    category: 'Electronics',
    price: 4199,
    demand: 'High',
    stock: 16,
  },
  {
    id: 4,
    name: 'Travel Backpack',
    category: 'Accessories',
    price: 1599,
    demand: 'Low',
    stock: 75,
  },
  {
    id: 5,
    name: 'Bluetooth Speaker',
    category: 'Electronics',
    price: 1899,
    demand: 'Medium',
    stock: 31,
  },
]

const demandData = [
  { day: 'Mon', value: 45 },
  { day: 'Tue', value: 68 },
  { day: 'Wed', value: 56 },
  { day: 'Thu', value: 82 },
  { day: 'Fri', value: 63 },
  { day: 'Sat', value: 96 },
  { day: 'Sun', value: 77 },
]

const formatCurrency = (value) =>
  new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(value)

const emptyProduct = {
  name: '',
  category: 'Electronics',
  price: '',
  demand: 'Medium',
  stock: '',
}

function App() {
  const [activePage, setActivePage] = useState('Overview')
  const [products, setProducts] = useState(initialProducts)
  const [search, setSearch] = useState('')
  const [period, setPeriod] = useState('7 days')
  const [showNotifications, setShowNotifications] = useState(false)
  const [showAddProduct, setShowAddProduct] = useState(false)
  const [newProduct, setNewProduct] = useState(emptyProduct)
  const [productMessage, setProductMessage] = useState('')
  const [apiAvailable, setApiAvailable] = useState(false)
  const [isSavingProduct, setIsSavingProduct] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function loadProducts() {
      try {
        const data = await getProducts()
        if (cancelled) return

        setProducts(data.map((product) => ({
          ...product,
          demand: product.demand || 'Medium',
          stock: Number.isInteger(product.stock) ? product.stock : 0,
        })))
        setApiAvailable(true)
        setProductMessage('')
      } catch (error) {
        if (!cancelled) {
          setApiAvailable(false)
          setProductMessage(`Backend unavailable; showing demo data. ${error.message}`)
        }
      }
    }

    loadProducts()
    return () => { cancelled = true }
  }, [])
  const filteredProducts = products.filter((product) =>
    `${product.name} ${product.category}`
      .toLowerCase()
      .includes(search.toLowerCase()),
  )

  const averagePrice = products.length
    ? Math.round(
        products.reduce((sum, product) => sum + product.price, 0) /
          products.length,
      )
    : 0

  const totalStock = products.reduce(
    (sum, product) => sum + product.stock,
    0,
  )

  const highDemandCount = products.filter(
    (product) => product.demand === 'High',
  ).length

  const updatePrice = async (id, amount) => {
    const existing = products.find((product) => product.id === id)
    if (!existing) return

    const updatedPrice = Math.max(1, Number(existing.price) + amount)
    setProductMessage('')

    if (apiAvailable) {
      try {
        await updateProduct(id, {
          name: existing.name,
          category: existing.category,
          price: updatedPrice,
          cost: Number(existing.cost ?? 0),
          description: existing.description ?? null,
        })
      } catch (error) {
        setProductMessage(`Price update failed: ${error.message}`)
        return
      }
    }

    setProducts((current) => current.map((product) =>
      product.id === id ? { ...product, price: updatedPrice } : product
    ))
    setProductMessage(`${existing.name} price updated.`)
  }
  const handleNewProductChange = (event) => {
    const { name, value } = event.target

    setNewProduct((current) => ({
      ...current,
      [name]: value,
    }))
  }

  const addProduct = async (event) => {
    event.preventDefault()
    setProductMessage('')

    const name = newProduct.name.trim()
    const price = Number(newProduct.price)
    const stock = Number(newProduct.stock)

    if (!name) {
      setProductMessage('Please enter a product name.')
      return
    }
    if (!Number.isFinite(price) || price <= 0) {
      setProductMessage('Price must be greater than zero.')
      return
    }
    if (!Number.isInteger(stock) || stock < 0) {
      setProductMessage('Stock must be a whole number of zero or more.')
      return
    }

    const payload = {
      name,
      category: newProduct.category,
      price,
      cost: 0,
      description: null,
    }

    setIsSavingProduct(true)
    try {
      let savedProduct

      if (apiAvailable) {
        savedProduct = await createProduct(payload)
      } else {
        savedProduct = { id: Date.now(), ...payload }
      }

      setProducts((current) => [
        ...current,
        { ...savedProduct, demand: newProduct.demand, stock },
      ])

      setNewProduct({ ...emptyProduct })
      setShowAddProduct(false)
      setProductMessage(
        apiAvailable
          ? `${name} saved to backend.`
          : `${name} added to demo data only; backend is unavailable.`,
      )
    } catch (error) {
      setProductMessage(`Could not save product: ${error.message}`)
    } finally {
      setIsSavingProduct(false)
    }
  }
  const cancelAddProduct = () => {
    setShowAddProduct(false)
    setNewProduct({ ...emptyProduct })
    setProductMessage('')
  }

  const exportReport = () => {
    const report = [
      ['Product', 'Category', 'Price (INR)', 'Demand', 'Stock'],
      ...products.map((product) => [
        product.name,
        product.category,
        product.price,
        product.demand,
        product.stock,
      ]),
    ]

    const csv = report
      .map((row) =>
        row
          .map((value) => `"${String(value).replaceAll('"', '""')}"`)
          .join(','),
      )
      .join('\n')

    const blob = new Blob(['\uFEFF' + csv], {
      type: 'text/csv;charset=utf-8;',
    })

    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')

    link.href = url
    link.download = 'econocausal-pricing-report.csv'
    document.body.appendChild(link)
    link.click()
    link.remove()

    URL.revokeObjectURL(url)
  }

  const navItems = [
    { name: 'Overview', icon: 'â—«' },
    { name: 'Products', icon: 'â–¦' },
    { name: 'Pricing Analysis', icon: 'â†—' },
    { name: 'DML Model', icon: 'â—Ž' },
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">E</div>

          <div>
            <h2>EconoCausal</h2>
            <span>INTELLIGENT PRICING</span>
          </div>
        </div>

        <p className="nav-heading">WORKSPACE</p>

        <nav className="navigation">
          {navItems.map((item) => (
            <button
              key={item.name}
              type="button"
              className={`nav-item ${
                activePage === item.name ? 'active' : ''
              }`}
              onClick={() => {
                setActivePage(item.name)
                setProductMessage('')
              }}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.name}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot" />

            <div>
              <strong>System status</strong>
              <span>Demo environment</span>
            </div>
          </div>

          <div className="profile">
            <div className="avatar">S</div>

            <div>
              <strong>Dashboard User</strong>
              <span>Analyst workspace</span>
            </div>

            <span className="profile-menu">â€¢â€¢â€¢</span>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span>/</span>
            <strong>{activePage}</strong>
          </div>

          <div className="topbar-actions">
            <span className="demo-badge">DEMO DATA</span>

            <div className="notification-wrapper">
              <button
                type="button"
                className="icon-button"
                aria-label="Notifications"
                onClick={() =>
                  setShowNotifications((current) => !current)
                }
              >
                â™§
              </button>

              {showNotifications && (
                <div className="notification-popover">
                  <strong>Notifications</strong>
                  <p>Your dashboard is running in demo mode.</p>
                  <p>Connect your backend to load live pricing data.</p>
                </div>
              )}
            </div>

            <div className="top-avatar">S</div>
          </div>
        </header>

        <div className="page-content">
          {/* OVERVIEW PAGE */}

          {activePage === 'Overview' && (
            <>
              <section className="page-heading">
                <div>
                  <p className="eyebrow">
                    DYNAMIC PRICING INTELLIGENCE
                  </p>

                  <h1>Overview</h1>

                  <p className="page-description">
                    Understand demand, evaluate pricing decisions, and
                    explore causal insights.
                  </p>
                </div>

                <button
                  type="button"
                  className="primary-button"
                  onClick={exportReport}
                >
                  â†“ Export report
                </button>
              </section>

              <section className="metrics-grid">
                <MetricCard
                  title="Products monitored"
                  value={products.length.toLocaleString('en-IN')}
                  change="Current catalog"
                  icon="â–¤"
                  color="purple"
                />

                <MetricCard
                  title="Average selling price"
                  value={formatCurrency(averagePrice)}
                  change="Sample product data"
                  icon="â‚¹"
                  color="blue"
                />

                <MetricCard
                  title="Total inventory"
                  value={totalStock.toLocaleString('en-IN')}
                  change="Units in catalog"
                  icon="â†—"
                  color="green"
                />

                <MetricCard
                  title="High-demand products"
                  value={highDemandCount}
                  change="Current catalog"
                  icon="â—Ž"
                  color="orange"
                />
              </section>

              <section className="overview-grid">
                <div className="panel demand-panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Demand overview</h3>
                      <p>Illustrative weekly demand index</p>
                    </div>

                    <select
                      value={period}
                      onChange={(event) => setPeriod(event.target.value)}
                      aria-label="Demand period"
                    >
                      <option value="7 days">Last 7 days</option>
                      <option value="30 days">Last 30 days</option>
                      <option value="90 days">Last 90 days</option>
                    </select>
                  </div>

                  <div className="demand-summary">
                    <strong>
                      {period === '7 days'
                        ? '72.4'
                        : period === '30 days'
                          ? '76.8'
                          : '69.5'}
                    </strong>

                    <span>Average demand index</span>
                  </div>

                  <div className="chart">
                    {demandData.map((item) => (
                      <div className="chart-column" key={item.day}>
                        <div className="bar-track">
                          <div
                            className="bar"
                            style={{
                              height: `${Math.min(
                                100,
                                item.value *
                                  (period === '30 days'
                                    ? 0.85
                                    : period === '90 days'
                                      ? 0.72
                                      : 1),
                              )}%`,
                            }}
                            title={`Demand index: ${item.value}`}
                          />
                        </div>

                        <span>{item.day}</span>
                      </div>
                    ))}
                  </div>

                  <div className="chart-legend">
                    <span />
                    Demand index (illustrative, not live data)
                  </div>
                </div>

                <div className="panel model-panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Causal model snapshot</h3>
                      <p>Double Machine Learning</p>
                    </div>

                    <span className="model-icon">â—Ž</span>
                  </div>

                  <span className="placeholder-badge">
                    DEMO PLACEHOLDER
                  </span>

                  <div className="coefficient">+0.084</div>

                  <p className="coefficient-description">
                    Example treatment effect coefficient
                  </p>

                  <div className="model-details">
                    <div>
                      <span>Model type</span>
                      <strong>Double ML</strong>
                    </div>

                    <div>
                      <span>Confidence interval</span>
                      <strong>Not calculated</strong>
                    </div>

                    <div>
                      <span>Last model run</span>
                      <strong>Not connected</strong>
                    </div>
                  </div>

                  <button
                    type="button"
                    className="text-button"
                    onClick={() => setActivePage('DML Model')}
                  >
                    Explore model details â†’
                  </button>
                </div>
              </section>
            </>
          )}

          {/* PRODUCTS PAGE */}

          {activePage === 'Products' && (
            <section className="page-section">
              <div className="page-heading">
                <div>
                  <p className="eyebrow">CATALOG MANAGEMENT</p>

                  <h1>Products</h1>

                  <p className="page-description">
                    Manage your catalog, search products, and adjust demo
                    prices.
                  </p>
                </div>

                <div className="topbar-actions">
                  <button
                    type="button"
                    className="primary-button"
                    onClick={() => {
                      setShowAddProduct((current) => !current)
                      setProductMessage('')
                    }}
                  >
                    {showAddProduct ? 'Cancel' : '+ Add Product'}
                  </button>

                  <button
                    type="button"
                    className="primary-button"
                    onClick={exportReport}
                  >
                    â†“ Export CSV
                  </button>
                </div>
              </div>

              {productMessage && (
                <div className="panel" role="status">
                  <p>{productMessage}</p>
                </div>
              )}

              {/* ADD PRODUCT FORM */}

              {showAddProduct && (
                <form
                  className="panel products-panel"
                  onSubmit={addProduct}
                >
                  <div className="panel-heading">
                    <div>
                      <h3>Add New Product</h3>
                      <p>
                        Enter the details below to add a product to the
                        catalog.
                      </p>
                    </div>
                  </div>

                  <div className="products-toolbar">
                    <input
                      type="text"
                      name="name"
                      placeholder="Product name"
                      value={newProduct.name}
                      onChange={handleNewProductChange}
                      maxLength={100}
                      required
                    />

                    <select
                      name="category"
                      value={newProduct.category}
                      onChange={handleNewProductChange}
                    >
                      <option value="Electronics">Electronics</option>
                      <option value="Fashion">Fashion</option>
                      <option value="Accessories">Accessories</option>
                      <option value="Home & Living">Home & Living</option>
                      <option value="Other">Other</option>
                    </select>

                    <input
                      type="number"
                      name="price"
                      placeholder="Price (â‚¹)"
                      min="1"
                      step="1"
                      value={newProduct.price}
                      onChange={handleNewProductChange}
                      required
                    />

                    <select
                      name="demand"
                      value={newProduct.demand}
                      onChange={handleNewProductChange}
                    >
                      <option value="High">High demand</option>
                      <option value="Medium">Medium demand</option>
                      <option value="Low">Low demand</option>
                    </select>

                    <input
                      type="number"
                      name="stock"
                      placeholder="Stock quantity"
                      min="0"
                      step="1"
                      value={newProduct.stock}
                      onChange={handleNewProductChange}
                      required
                    />

                    <button
                      type="submit"
                      className="primary-button"
                    >
                      Save Product
                    </button>

                    <button
                      type="button"
                      className="text-button"
                      onClick={cancelAddProduct}
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              )}

              {/* PRODUCT TABLE */}

              <div className="panel products-panel">
                <div className="products-toolbar">
                  <h3>Product catalog ({products.length})</h3>

                  <input
                    type="search"
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    placeholder="Search products..."
                    aria-label="Search products"
                  />
                </div>

                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>PRODUCT</th>
                        <th>CATEGORY</th>
                        <th>PRICE</th>
                        <th>DEMAND</th>
                        <th>STOCK</th>
                        <th>DEMO PRICE ACTIONS</th>
                      </tr>
                    </thead>

                    <tbody>
                      {filteredProducts.map((product) => (
                        <tr key={product.id}>
                          <td className="product-name">
                            {product.name}
                          </td>

                          <td>{product.category}</td>

                          <td>{formatCurrency(product.price)}</td>

                          <td>
                            <span
                              className={`demand-tag ${product.demand.toLowerCase()}`}
                            >
                              {product.demand}
                            </span>
                          </td>

                          <td>{product.stock}</td>

                          <td>
                            <div className="price-actions">
                              <button
                                type="button"
                                onClick={() =>
                                  updatePrice(product.id, -100)
                                }
                                aria-label={`Decrease ${product.name} price`}
                                title="Decrease price by â‚¹100"
                              >
                                âˆ’
                              </button>

                              <button
                                type="button"
                                onClick={() =>
                                  updatePrice(product.id, 100)
                                }
                                aria-label={`Increase ${product.name} price`}
                                title="Increase price by â‚¹100"
                              >
                                +
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>

                  {filteredProducts.length === 0 && (
                    <p className="empty-state">
                      No matching products found.
                    </p>
                  )}
                </div>

                <p className="demo-note">
                  Products and price changes currently use local demo
                  state. They are not saved to a backend or database.
                </p>
              </div>
            </section>
          )}

          {/* PRICING ANALYSIS PAGE */}

          {activePage === 'Pricing Analysis' && (
            <section className="page-section">
              <div className="page-heading">
                <div>
                  <p className="eyebrow">PRICE &amp; DEMAND</p>

                  <h1>Pricing Analysis</h1>

                  <p className="page-description">
                    Explore sample price points and demand patterns.
                  </p>
                </div>

                <button
                  type="button"
                  className="primary-button"
                  onClick={exportReport}
                >
                  â†“ Export CSV
                </button>
              </div>

              <div className="analysis-grid">
                <div className="panel analysis-card">
                  <span className="analysis-label">
                    AVERAGE CATALOG PRICE
                  </span>

                  <strong>{formatCurrency(averagePrice)}</strong>

                  <p>Calculated from the current product catalog.</p>
                </div>

                <div className="panel analysis-card">
                  <span className="analysis-label">
                    HIGH-DEMAND PRODUCTS
                  </span>

                  <strong>{highDemandCount}</strong>

                  <p>Products marked as high demand in demo data.</p>
                </div>

                <div className="panel analysis-card">
                  <span className="analysis-label">
                    TOTAL SAMPLE INVENTORY
                  </span>

                  <strong>{totalStock}</strong>

                  <p>Units across the current product catalog.</p>
                </div>
              </div>

              <div className="panel products-panel">
                <h3>Sample pricing recommendations</h3>

                <div className="recommendation-list">
                  {products.map((product) => (
                    <div
                      className="recommendation-row"
                      key={product.id}
                    >
                      <div>
                        <strong>{product.name}</strong>

                        <p>
                          {product.demand === 'High'
                            ? 'Review demand and stock before changing price.'
                            : product.demand === 'Low'
                              ? 'Consider a promotion and monitor conversion.'
                              : 'Compare performance with similar products.'}
                        </p>
                      </div>

                      <strong>{formatCurrency(product.price)}</strong>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          )}

          {/* DML MODEL PAGE */}

          {activePage === 'DML Model' && (
            <section className="page-section">
              <div className="page-heading">
                <div>
                  <p className="eyebrow">CAUSAL INFERENCE</p>

                  <h1>Double Machine Learning</h1>

                  <p className="page-description">
                    Understand the model used to estimate causal treatment
                    effects in pricing.
                  </p>
                </div>
              </div>

              <div className="panel dml-intro">
                <span className="placeholder-badge">
                  BACKEND NOT CONNECTED
                </span>

                <h2>What is Double Machine Learning?</h2>

                <p>
                  Double Machine Learning (DML) uses machine-learning
                  models to account for observed factors that affect both
                  pricing decisions and outcomes. It can help estimate
                  the causal effect of a price or treatment while
                  reducing bias from observed confounders, under suitable
                  assumptions.
                </p>
              </div>

              <div className="analysis-grid">
                <div className="panel analysis-card">
                  <span className="analysis-label">TREATMENT</span>

                  <strong>Price</strong>

                  <p>
                    The pricing variable whose effect we want to estimate.
                  </p>
                </div>

                <div className="panel analysis-card">
                  <span className="analysis-label">OUTCOME</span>

                  <strong>Demand</strong>

                  <p>
                    The business outcome the model seeks to explain.
                  </p>
                </div>

                <div className="panel analysis-card">
                  <span className="analysis-label">ESTIMATE</span>

                  <strong>Not available</strong>

                  <p>
                    Connect the trained model API to display real estimates.
                  </p>
                </div>
              </div>

              <div className="panel model-warning">
                <strong>Important</strong>

                <p>
                  No real causal estimate is calculated by this frontend.
                  Connect the backend model and its confidence intervals
                  before using results for business decisions.
                </p>
              </div>
            </section>
          )}

          <footer className="page-footer">
            <span>EconoCausal Â· Intelligent Pricing</span>
            <span>Frontend prototype Â· Sample data only</span>
          </footer>
        </div>
      </main>
    </div>
  )
}

function MetricCard({ title, value, change, icon, color }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{title}</span>

        <div className={`metric-icon ${color}`}>
          {icon}
        </div>
      </div>

      <strong className="metric-value">{value}</strong>

      <span className="metric-change">{change}</span>
    </div>
  )
}

export default App

