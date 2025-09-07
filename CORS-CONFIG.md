# CORS Configuration

The RAG service uses regex-based CORS origin matching for flexible domain configuration.

## Environment Variables

- `CORS_ORIGIN_REGEX`: **Required** - Regular expression pattern to match allowed origins

## Usage Examples

### Local Development
```bash
# Matches: http(s)://anything.docker.localhost with optional port OR localhost with port
CORS_ORIGIN_REGEX=^https?://((.*\.)?docker\.localhost(:[0-9]+)?|localhost(:[0-9]+)?)$
```

### Production
```bash
# Matches: https://lokotest.pl and https://anything.lokotest.pl
CORS_ORIGIN_REGEX=^https://(.*\.)?lokotest\.pl$
```

## Pattern Examples

### Local Development Patterns
- `^https?://(.*\.)?docker\.localhost(:[0-9]+)?$` - Any subdomain of docker.localhost with optional port
- `^https?://localhost(:[0-9]+)?$` - localhost with optional port
- `^https?://((.*\.)?docker\.localhost(:[0-9]+)?|localhost(:[0-9]+)?)$` - Combined pattern

### Production Patterns  
- `^https://(.*\.)?lokotest\.pl$` - lokotest.pl and all subdomains (HTTPS only)
- `^https://lokotest\.pl$` - Only main domain (no subdomains)

## Testing CORS

```bash
# Test preflight request
curl -i -X OPTIONS \
  -H "Origin: http://loko.docker.localhost:4600" \
  -H "Access-Control-Request-Method: GET" \
  http://rag.docker.localhost/ask

# Test actual request
curl -i -H "Origin: http://loko.docker.localhost:4600" \
  "http://rag.docker.localhost/ask?q=test"
```