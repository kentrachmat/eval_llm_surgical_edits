-- strip_elements.lua
-- Pandoc Lua filter to remove Math, Tables, and Figures.

function Math(el)
  return {} -- Remove Math elements (both inline and display)
end

function Table(el)
  return {} -- Remove Table elements
end

function Figure(el)
  return {} -- Remove Figure elements
end

-- For more granular control, you could define for specific types:
-- function InlineMath(el) return {} end
-- function DisplayMath(el) return {} end

-- The filter will apply these functions to the respective element types.
-- If a function for an element type is not defined, the element is passed through unchanged.