--
-- PostgreSQL database dump
--

-- Dumped from database version 16.2
-- Dumped by pg_dump version 16.2

-- Started on 2024-12-19 18:13:15

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 229 (class 1259 OID 122348)
-- Name: comp_esco; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.comp_esco (
    id integer NOT NULL,
    uri text NOT NULL,
    skill_name text NOT NULL,
    embedding double precision[]
);


ALTER TABLE public.comp_esco OWNER TO postgres;

--
-- TOC entry 228 (class 1259 OID 122347)
-- Name: comp_esco_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.comp_esco_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.comp_esco_id_seq OWNER TO postgres;

--
-- TOC entry 4848 (class 0 OID 0)
-- Dependencies: 228
-- Name: comp_esco_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.comp_esco_id_seq OWNED BY public.comp_esco.id;


--
-- TOC entry 221 (class 1259 OID 24428)
-- Name: encoded_segments; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.encoded_segments (
    id integer NOT NULL,
    segment text NOT NULL,
    embedding double precision[] NOT NULL
);


ALTER TABLE public.encoded_segments OWNER TO postgres;

--
-- TOC entry 220 (class 1259 OID 24427)
-- Name: encoded_segments_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.encoded_segments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.encoded_segments_id_seq OWNER TO postgres;

--
-- TOC entry 4849 (class 0 OID 0)
-- Dependencies: 220
-- Name: encoded_segments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.encoded_segments_id_seq OWNED BY public.encoded_segments.id;


--
-- TOC entry 215 (class 1259 OID 23935)
-- Name: offres_emploi; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.offres_emploi (
    ref_offre text,
    descri_mission text
);


ALTER TABLE public.offres_emploi OWNER TO postgres;

--
-- TOC entry 231 (class 1259 OID 136605)
-- Name: offres_extract; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.offres_extract (
    id integer NOT NULL,
    id_offre character varying NOT NULL,
    intitule character varying NOT NULL,
    description text
);


ALTER TABLE public.offres_extract OWNER TO postgres;

--
-- TOC entry 230 (class 1259 OID 136604)
-- Name: offres_extract_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.offres_extract_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.offres_extract_id_seq OWNER TO postgres;

--
-- TOC entry 4850 (class 0 OID 0)
-- Dependencies: 230
-- Name: offres_extract_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.offres_extract_id_seq OWNED BY public.offres_extract.id;


--
-- TOC entry 219 (class 1259 OID 24406)
-- Name: table_comp; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.table_comp (
    id integer NOT NULL,
    ref_offre character varying,
    segment character varying,
    comp_predict integer
);


ALTER TABLE public.table_comp OWNER TO postgres;

--
-- TOC entry 218 (class 1259 OID 24405)
-- Name: table_comp_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.table_comp_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.table_comp_id_seq OWNER TO postgres;

--
-- TOC entry 4851 (class 0 OID 0)
-- Dependencies: 218
-- Name: table_comp_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.table_comp_id_seq OWNED BY public.table_comp.id;


--
-- TOC entry 217 (class 1259 OID 24397)
-- Name: table_contxt; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.table_contxt (
    id integer NOT NULL,
    ref_offre character varying,
    segment character varying,
    offre_predict integer
);


ALTER TABLE public.table_contxt OWNER TO postgres;

--
-- TOC entry 216 (class 1259 OID 24396)
-- Name: table_contxt_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.table_contxt_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.table_contxt_id_seq OWNER TO postgres;

--
-- TOC entry 4852 (class 0 OID 0)
-- Dependencies: 216
-- Name: table_contxt_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.table_contxt_id_seq OWNED BY public.table_contxt.id;


--
-- TOC entry 227 (class 1259 OID 24544)
-- Name: table_monitoring_comp; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.table_monitoring_comp (
    id integer NOT NULL,
    ref_user character varying,
    segment character varying,
    prediction integer,
    feedback_user character varying,
    embedding double precision[]
);


ALTER TABLE public.table_monitoring_comp OWNER TO postgres;

--
-- TOC entry 226 (class 1259 OID 24543)
-- Name: table_monitoring_comp_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.table_monitoring_comp_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.table_monitoring_comp_id_seq OWNER TO postgres;

--
-- TOC entry 4853 (class 0 OID 0)
-- Dependencies: 226
-- Name: table_monitoring_comp_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.table_monitoring_comp_id_seq OWNED BY public.table_monitoring_comp.id;


--
-- TOC entry 225 (class 1259 OID 24535)
-- Name: table_monitoring_contxt; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.table_monitoring_contxt (
    id integer NOT NULL,
    ref_user character varying,
    segment character varying,
    prediction_contxt integer,
    feedback_user character varying,
    embedding double precision[]
);


ALTER TABLE public.table_monitoring_contxt OWNER TO postgres;

--
-- TOC entry 224 (class 1259 OID 24534)
-- Name: table_monitoring_contxt_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.table_monitoring_contxt_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.table_monitoring_contxt_id_seq OWNER TO postgres;

--
-- TOC entry 4854 (class 0 OID 0)
-- Dependencies: 224
-- Name: table_monitoring_contxt_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.table_monitoring_contxt_id_seq OWNED BY public.table_monitoring_contxt.id;


--
-- TOC entry 223 (class 1259 OID 24488)
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying(50) NOT NULL,
    password_hash character varying(255) NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.users OWNER TO postgres;

--
-- TOC entry 222 (class 1259 OID 24487)
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO postgres;

--
-- TOC entry 4855 (class 0 OID 0)
-- Dependencies: 222
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- TOC entry 4680 (class 2604 OID 122351)
-- Name: comp_esco id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.comp_esco ALTER COLUMN id SET DEFAULT nextval('public.comp_esco_id_seq'::regclass);


--
-- TOC entry 4675 (class 2604 OID 24431)
-- Name: encoded_segments id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.encoded_segments ALTER COLUMN id SET DEFAULT nextval('public.encoded_segments_id_seq'::regclass);


--
-- TOC entry 4681 (class 2604 OID 136608)
-- Name: offres_extract id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.offres_extract ALTER COLUMN id SET DEFAULT nextval('public.offres_extract_id_seq'::regclass);


--
-- TOC entry 4674 (class 2604 OID 24409)
-- Name: table_comp id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.table_comp ALTER COLUMN id SET DEFAULT nextval('public.table_comp_id_seq'::regclass);


--
-- TOC entry 4673 (class 2604 OID 24400)
-- Name: table_contxt id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.table_contxt ALTER COLUMN id SET DEFAULT nextval('public.table_contxt_id_seq'::regclass);


--
-- TOC entry 4679 (class 2604 OID 24547)
-- Name: table_monitoring_comp id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.table_monitoring_comp ALTER COLUMN id SET DEFAULT nextval('public.table_monitoring_comp_id_seq'::regclass);


--
-- TOC entry 4678 (class 2604 OID 24538)
-- Name: table_monitoring_contxt id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.table_monitoring_contxt ALTER COLUMN id SET DEFAULT nextval('public.table_monitoring_contxt_id_seq'::regclass);


--
-- TOC entry 4676 (class 2604 OID 24491)
-- Name: users id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- TOC entry 4697 (class 2606 OID 122355)
-- Name: comp_esco comp_esco_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.comp_esco
    ADD CONSTRAINT comp_esco_pkey PRIMARY KEY (id);


--
-- TOC entry 4687 (class 2606 OID 24435)
-- Name: encoded_segments encoded_segments_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.encoded_segments
    ADD CONSTRAINT encoded_segments_pkey PRIMARY KEY (id);


--
-- TOC entry 4699 (class 2606 OID 136612)
-- Name: offres_extract offres_extract_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.offres_extract
    ADD CONSTRAINT offres_extract_pkey PRIMARY KEY (id);


--
-- TOC entry 4685 (class 2606 OID 24413)
-- Name: table_comp table_comp_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.table_comp
    ADD CONSTRAINT table_comp_pkey PRIMARY KEY (id);


--
-- TOC entry 4683 (class 2606 OID 24404)
-- Name: table_contxt table_contxt_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.table_contxt
    ADD CONSTRAINT table_contxt_pkey PRIMARY KEY (id);


--
-- TOC entry 4695 (class 2606 OID 24551)
-- Name: table_monitoring_comp table_monitoring_comp_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.table_monitoring_comp
    ADD CONSTRAINT table_monitoring_comp_pkey PRIMARY KEY (id);


--
-- TOC entry 4693 (class 2606 OID 24542)
-- Name: table_monitoring_contxt table_monitoring_contxt_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.table_monitoring_contxt
    ADD CONSTRAINT table_monitoring_contxt_pkey PRIMARY KEY (id);


--
-- TOC entry 4689 (class 2606 OID 24494)
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- TOC entry 4691 (class 2606 OID 24496)
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


-- Completed on 2024-12-19 18:13:15

--
-- PostgreSQL database dump complete
--

